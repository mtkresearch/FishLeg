open Printf
open Base
open Owl
open Fishleg

module Net = Network.MLP (struct
  let activation_fun ~layer = if layer = 3 || layer = 7 then Fn.id else AD.Maths.relu
end)

module Lik = Likelihood.Bernoulli
module F = Approx.Block_Kronecker
module M = Fishleg.Model (Net) (F) (Lik)

let lrate = Setup.LR.constant
let update_lambda_every = Cmdargs.(get_int "-update_lambda_every" |> default 10)

module S =
  Setup.Make
    (M)
    (struct
      let eta_sgd_ref = 0.03
      let eta_fl_ref = 0.01
      let eta_sgd = lrate Cmdargs.(get_float "-eta_sgd" |> default 3E-5)
      let eta_sgdm = lrate Cmdargs.(get_float "-eta_sgdm" |> default 0.03)
      let eta_adam = lrate Cmdargs.(get_float "-eta_adam" |> default 1E-4)
      let eta_fl = lrate Cmdargs.(get_float "-eta_fl" |> default eta_fl_ref)

      let eta_fl_aux k =
        if k % update_lambda_every = 0
        then lrate Cmdargs.(get_float "-eta_fl_aux" |> default 1e-3) k
        else None

      let bs = 100
      let bs_curvature = 10
      let beta = 0.9
      let weight_decay = Some 1e-5
      let aux_weight_decay = Some Cmdargs.(get_float "-aux_weight_decay" |> default 0.001)
      let damping = None
      let n_aux_iter _ = 1
      let save_loss_every = Some 10
      let save_lambda_every = None
    end)

let _ =
  Bos.Cmd.(v "cp" % "-f" % "src/mn.ml" % S.in_dir "mn.ml" |> Bos.OS.Cmd.run) |> ignore

let n = 784
let sizes = [| n; 1000; 500; 250; 30; 250; 500; 1000; n |]

let set typ =
  let x = Owl.Mat.load_npy "_data/mnist_x.npy" in
  let slice =
    match typ with
    | `train -> [ 0; 60_000 - 1 ]
    | `test -> [ 60_000; -1 ]
  in
  Arr.get_slice [ slice ] x, Arr.get_slice [ slice ] x

let train_set = set `train
let test_set = set `test

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
  if Cmdargs.check "-reuse"
  then M.W.load S.(in_dir "../w.bin")
  else { net = Net.init sizes; lik = () }

let lambda () : M.L.t =
  let net = F.init_diag ~with_bias:true ~scale:S.scale sizes in
  let lik = () in
  { net; lik }

module One_run () = struct
  let compute_mse set w =
    let w = M.W.value w in
    let data_x, data_y = sample_data set 10000 in
    let y_pred = Net.f ~w:w.net data_x in
    Lik.mean_squared_error ~obs:data_y y_pred

  let things_to_log = 100, [ compute_mse train_set; compute_mse test_set ]

  let losses =
    try
      let w_opt, losses =
        S.train ~things_to_log ~w:(w ()) ~lambda:(lambda ()) (sample_data train_set)
      in
      if Cmdargs.check "-save" then M.W.save ~out:(S.in_dir "w.bin") w_opt;
      Some losses
    with
    | Setup.Nan_loss -> None
end

(* perform multiple runs *)
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
