open Base
open Owl
include Fishleg_typ

module Model
  (Net : Network.T)
  (F : Approx.T with type 'a W.p = 'a Net.W.p)
  (Lik : Likelihood.T with type stats = Net.output) =
struct
  open P
  module Net = Net
  module F = F
  module Lik = Lik
  module W = P.Make (Net.W) (Lik.W)
  module L = P.Make (F.L) (Lik.L)

  type input = Net.input
  type output = Lik.data

  let sample ~w input =
    let z = Net.f ~w:w.net input in
    fun () -> input, Lik.sample ~w:w.lik z

  let neg_logp ~data w =
    let input, output = data in
    let y_pred = Net.f ~w:w.net input in
    Lik.neg_logp ~w:w.lik ~obs:output y_pred

  let cross_entropy =
    match Lik.expected_neg_logp with
    | None ->
      fun ~w ~input delta ->
        let data = sample ~w input () in
        let w' = W.(w + delta) in
        neg_logp ~data w'
    | Some enll ->
      fun ~w ~input delta ->
        let expectation_under = Net.f ~w:w.net input, w.lik in
        let w' = W.(w + delta) in
        enll ~expectation_under (Net.f ~w:w'.net input, w'.lik)

  let q_v_prod ~lambda u =
    { net = F.q_v_prod ~lambda:lambda.net u.net
    ; lik = Lik.q_v_prod ~lambda:lambda.lik u.lik
    }

  let aux_value_and_grad ?damping ~input ~w ~g lambda =
    let g2 = W.dot_prod g g in
    (* to normalize the aux loss *)
    let lambda = L.make_reverse lambda (AD.tag ()) in
    let zz = q_v_prod ~lambda g in
    (* take this off the AD tape as we will construct the gradient more efficiently by hand *)
    let z = W.map zz ~f:AD.primal in
    (* perform Hessian vector product of the cross-entropy at delta=0 with v=z *)
    let hz =
      let delta = W.(map ~f:Prms.free (zeros_of z)) in
      let hz = W.hessian_v ~f:(cross_entropy ~w ~input) ~v:z delta in
      match damping with
      | None -> hz
      | Some d -> W.(hz + (F d $* z))
    in
    let hzmg = W.(hz - g) in
    let aux_loss =
      AD.Maths.(F 0.5 * (W.dot_prod hzmg z - W.dot_prod g z) / g2) |> AD.unpack_flt
    in
    (* compute the gradient w.r.t lambda *)
    let grad_lambda =
      let loss = AD.Maths.(W.dot_prod hzmg zz / g2) in
      AD.reverse_prop (F 1.) loss;
      L.map lambda ~f:AD.adjval
    in
    aux_loss, grad_lambda

  let aux_loop ~config sample_data =
    let module OL = Prms.Opt.Adam.Make (L) in
    fun ~k ~w ~g (lambda, state) ->
      let g = g () in
      match config.learning_rate_aux k with
      | Some eta ->
        let adam_config = { config.adam with learning_rate = Some eta } in
        let w = W.value w in
        let rec iter i accu (lambda, state) =
          if i > config.n_aux_iter k
          then List.rev accu, (lambda, state)
          else (
            let input, _ = sample_data config.bs_curvature in
            let aux_loss, delta =
              aux_value_and_grad ?damping:config.damping ~input ~w ~g lambda
            in
            let lambda, state = OL.step ~config:adam_config ?state ~delta lambda in
            iter (i + 1) (aux_loss :: accu) (lambda, state))
        in
        iter 1 [] (lambda, state)
      | None -> [ 0. ], (lambda, state)

  let value_and_grad_with_info ~data ~w =
    match Net.manual_f_df with
    | None -> W.value_and_grad ~f:(neg_logp ~data) w
    | Some mfdf ->
      let w = W.value w in
      let input, output = data in
      let tag = AD.tag () in
      let w_lik = Lik.W.map w.lik ~f:(fun x -> AD.make_reverse x tag) in
      let loss_fun y_pred = Lik.neg_logp ~w:w_lik ~obs:output y_pred in
      let loss, dw = mfdf ~tag ~w:w.net input loss_fun in
      loss, { net = dw; lik = Lik.W.map w_lik ~f:AD.adjval }

  let train ?monitor ~config ~w ~lambda ~n_iter sample_data =
    let module OW = Prms.Opt.SGD.Make (W) in
    let aux_loop = aux_loop ~config sample_data in
    let rec iter ~k ~w ~lambda ~state ~ngbar =
      Stdlib.Gc.major ();
      (* start by updating the aux function *)
      let ell, g = value_and_grad_with_info ~data:(sample_data config.bs) ~w in
      let g_for_aux () = g in
      (* start by updating the aux function *)
      let aux_losses, (lambda, state) = aux_loop ~k ~w ~g:g_for_aux (lambda, state) in
      let aux_loss = aux_losses |> Array.of_list |> Stats.mean in
      Option.iter monitor ~f:(fun m ->
        m ~k ~loss:(AD.unpack_flt ell) ~aux_loss ~g (w, lambda));
      (* approximate Natural Gradient step *)
      let nat_grad = q_v_prod ~lambda:(L.value lambda) g in
      let ngbar =
        W.((AD.F (1. -. config.beta) $* nat_grad) + (AD.F config.beta $* ngbar))
      in
      let w, _ =
        let learning_rate = config.learning_rate k in
        OW.step
          ~config:{ learning_rate; weight_decay = config.weight_decay }
          ~delta:ngbar
          w
      in
      (* iterate unless finished... *)
      if k < n_iter then iter ~k:(k + 1) ~w ~lambda ~state ~ngbar else w
    in
    (* start the loop *)
    let ngbar = W.(zeros_of (W.value w)) in
    iter ~k:0 ~w ~lambda ~state:None ~ngbar

  let train_sgd ?monitor ~config ~learning_rate ~bs ~w ~n_iter sample_data =
    let module OW = Prms.Opt.SGD.Make (W) in
    let rec iter ~k w =
      if k > n_iter
      then w
      else (
        Stdlib.Gc.major ();
        let ell, g = W.value_and_grad ~f:(neg_logp ~data:(sample_data bs)) w in
        let w, _ =
          OW.step ~config:{ config with learning_rate = learning_rate k } ~delta:g w
        in
        Option.iter monitor ~f:(fun m -> m ~k ~loss:(AD.unpack_flt ell) ~aux_loss:0. ~g w);
        iter ~k:(k + 1) w)
    in
    iter ~k:0 w

  let train_sgdm ?monitor ~config ~learning_rate ~bs ~w ~n_iter sample_data =
    let module OW = Prms.Opt.SGD_momentum.Make (W) in
    let rec iter ~k w =
      if k > n_iter
      then w
      else (
        Stdlib.Gc.major ();
        let ell, g = W.value_and_grad ~f:(neg_logp ~data:(sample_data bs)) w in
        let w, _ =
          OW.step ~config:{ config with learning_rate = learning_rate k } ~delta:g w
        in
        Option.iter monitor ~f:(fun m -> m ~k ~loss:(AD.unpack_flt ell) ~aux_loss:0. ~g w);
        iter ~k:(k + 1) w)
    in
    iter ~k:0 w

  let train_adam ?monitor ~config ~learning_rate ~bs ~w ~n_iter sample_data =
    let module OW = Prms.Opt.Adam.Make (W) in
    let rec iter ~k (w, state) =
      if k > n_iter
      then w
      else (
        Stdlib.Gc.major ();
        let ell, g = W.value_and_grad ~f:(neg_logp ~data:(sample_data bs)) w in
        let w, state =
          OW.step
            ~config:{ config with learning_rate = learning_rate k }
            ?state
            ~delta:g
            w
        in
        Option.iter monitor ~f:(fun m -> m ~k ~loss:(AD.unpack_flt ell) ~aux_loss:0. ~g w);
        iter ~k:(k + 1) (w, state))
    in
    iter ~k:0 (w, None)
end
