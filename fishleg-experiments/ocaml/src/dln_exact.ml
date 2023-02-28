open Printf
open Base
open Owl
open Fishleg
module Net = Network.MLP_linear_no_bias
module Lik = Likelihood.Gaussian
module F = Approx.Block_Kronecker
module M = Fishleg.Model (Net) (F) (Lik)

let lrate = Setup.LR.constant

module S =
  Setup.Make
    (M)
    (struct
      let eta_fl_ref = 0.04
      let eta_sgd_ref = 1E-3
      let eta_sgd = lrate Cmdargs.(get_float "-eta_sgd" |> default 3E-5)
      let eta_sgdm = lrate Cmdargs.(get_float "-eta_sgdm" |> default 2E-3)
      let eta_adam = lrate Cmdargs.(get_float "-eta_adam" |> default 2E-4)
      let eta_fl = lrate eta_fl_ref
      let eta_fl_aux = lrate Cmdargs.(get_float "-eta_fl_aux" |> default 1e-3)
      let bs = 40
      let bs_curvature = 40
      let beta = 0.9
      let weight_decay = None
      let aux_weight_decay = Some Cmdargs.(get_float "-aux_weight_decay" |> default 1e-4)
      let damping = None
      let n_aux_iter _ = 1
      let save_loss_every = Some 100
      let save_lambda_every = None
    end)

let _ =
  if C.first && Poly.(S.algo = `fl)
  then ignore (Bos.Cmd.(v "rm" % "-f" % S.in_dir "fl_aux_tracking") |> Bos.OS.Cmd.run)

let n = 100
let n_layers = 20
let size = 20
let sizes = Array.create ~len:n_layers size
let sigma_true = 1E-3
let sigma_init = 0.1

let sample_data (set_x, set_y) =
  let a = Mat.row_num set_x in
  fun batch_size ->
    if batch_size < 0
    then AD.(pack_arr set_x, pack_arr set_y)
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let xs = AD.pack_arr (Mat.get_fancy [ L ids ] set_x) in
      let ys = AD.pack_arr (Mat.get_fancy [ L ids ] set_y) in
      xs, ys)

(* initialisation of network parameters *)
let w () : M.W.t =
  { net = Net.init sizes
  ; lik = Prms.create ~above:(AD.F 1E-4) ~below:(AD.F 1.) (AD.F sigma_init)
  }

let lambda () : M.L.t =
  let scale = S.scale in
  let net = F.init_diag ~with_bias:false ~scale sizes in
  let lik = Lik.init_lambda ~scale in
  { net; lik }

(* exact computation of the inverse Fisher for each block
   (and we know it's exact to consider only diag blocks *)

let behind_matrices (w : Net.W.t') =
  let open Network.MLP_Layer_P in
  w
  |> Array.fold
       ~init:[ AD.Mat.eye sizes.(0) ]
       ~f:(fun accu layer -> AD.Maths.(List.hd_exn accu *@ layer.w) :: accu)
  |> List.rev
  |> Array.of_list
  |> Array.map ~f:AD.unpack_arr

let ahead_matrices (w : Net.W.t') =
  let open Network.MLP_Layer_P in
  w
  |> Array.rev
  |> Array.fold
       ~init:[ AD.Mat.eye (Array.last sizes) ]
       ~f:(fun accu layer -> AD.Maths.(layer.w *@ List.hd_exn accu) :: accu)
  |> Array.of_list
  |> Array.map ~f:AD.unpack_arr

let fisher_v_prod ~(w : M.W.t') v =
  let open Network.MLP_Layer_P in
  let b = behind_matrices w.net in
  let a = ahead_matrices w.net in
  let sigma = AD.unpack_flt w.lik in
  (* go through each block of the Fisher and compute products
     using a Kronecker identity *)
  Array.init (Array.length v) ~f:(fun i ->
    let wbi = b.(i)
    and wai = a.(i + 1) in
    Array.foldi v ~init:None ~f:(fun j accu vj ->
      let vj = AD.unpack_arr vj.w in
      let wbj = b.(j)
      and waj = a.(j + 1) in
      let bf = Mat.(transpose wbi *@ wbj) in
      let af = Mat.(waj *@ transpose wai /$ Float.(square sigma)) in
      let z = Mat.(bf *@ vj *@ af) in
      match accu with
      | Some a -> Some Mat.(a + z)
      | None -> Some z)
    |> Option.value_exn
    |> AD.pack_arr)

module One_run () = struct
  let teacher : M.W.t = { net = Net.init sizes; lik = Lik.init sigma_true }
  let teacher' = M.W.value teacher

  let set kind =
    let k =
      match kind with
      | `train -> 40
      | `test -> 40
    in
    let xs = AD.Arr.gaussian [| k; sizes.(0) |] in
    let _, ys = M.sample ~w:teacher' xs () in
    AD.unpack_arr xs, AD.unpack_arr ys

  let train_set = set `train
  let test_set = set `test

  let compute_loss set w =
    let data = sample_data set (-1) in
    fst (M.value_and_grad_with_info ~data ~w) |> AD.unpack_flt

  let sample_data = sample_data train_set
  let lambda_frozen = Array.create ~len:10 None

  (* periodically check the auxiliary loss for frozen parameters *)
  let lambda_hook ~k ~(w : M.W.t) ~g:_ ~lambda =
    if k % 500 = 0
    then (
      try lambda_frozen.(k / 500) <- Some lambda with
      | _ -> ());
    if k % 100 = 0
    then (
      (* compute on whole train set to get lower-variance estimates *)
      let _, g = M.value_and_grad_with_info ~data:(sample_data S.bs) ~w in
      (* renormalise g *)
      let gtilde =
        let g_norm = M.W.dot_prod g g |> AD.Maths.sqrt in
        M.W.map g ~f:(fun x -> AD.Maths.(x / g_norm))
      in
      let w = M.W.value w in
      let aux_loss lambda =
        let lambda = M.L.value lambda in
        let z = (M.q_v_prod ~lambda gtilde).net in
        let fz =
          fisher_v_prod ~w z
          |> Array.map ~f:(fun x -> Network.MLP_Layer_P.{ w = x; grad_info = None })
        in
        let diff = Net.W.(F 0.5 $* fz - gtilde.net) in
        Net.W.dot_prod diff z |> AD.unpack_flt
      in
      let results =
        Mat.of_arrays
          [| Array.append
               [| Float.of_int k; aux_loss lambda |]
               (Array.map lambda_frozen ~f:(function
                 | None -> -10000000.
                 | Some lam -> aux_loss lam))
          |]
        |> C.gather
      in
      if C.first
      then
        Mat.(
          save_txt
            ~append:true
            ~out:(S.in_dir "fl_aux_tracking")
            (mean ~axis:0 (concatenate ~axis:0 results))))

  let extract_sigma (w : M.W.t) = Prms.value w.lik |> AD.unpack_flt

  let things_to_log =
    ( 100
    , [ compute_loss train_set
      ; compute_loss test_set
      ; (fun _ -> compute_loss test_set teacher)
      ; extract_sigma
      ] )

  let losses =
    try
      let _, losses =
        S.train ~lambda_hook ~things_to_log ~w:(w ()) ~lambda:(lambda ()) sample_data
      in
      Some losses
    with
    | Setup.Nan_loss -> None
end

(* let MPI perform multiple runs *)
let () =
  let module R = One_run () in
  let runs = C.gather R.losses in
  if C.first
  then (
    let runs = Array.filter_opt runs in
    let n_failed = C.n_nodes - Array.length runs in
    Bos.Cmd.(
      v "touch" % S.in_dir (sprintf "%s_%i_failed" S.info_file n_failed) |> Bos.OS.Cmd.run)
    |> ignore;
    if Array.length runs > 0
    then (
      let runs = Array.map ~f:(fun x -> Arr.expand x 3) runs |> Arr.concatenate ~axis:0 in
      let stats = Arr.(mean ~axis:0 runs @|| std ~axis:0 runs) |> Arr.squeeze in
      Mat.save_txt ~out:(S.in_dir (sprintf "%s_stats" S.info_file)) stats;
      Mat.save ~out:(S.in_dir (sprintf "%s_stats.bin" S.info_file)) stats))
