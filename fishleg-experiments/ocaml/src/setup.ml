open Base
open Fishleg

exception Nan_loss

module LR = struct
  type t = int -> float option

  let init_exn lr =
    match lr 0 with
    | Some eta -> eta
    | None -> assert false

  let none (_ : int) = None
  let constant eta (_ : int) = Some eta

  let cosine ~period ~eta_min ~eta_max k =
    let k = k % period in
    Some
      Float.(
        eta_min + (0.5 * (eta_max - eta_min) * (1. + cos (of_int k * pi / of_int period))))

  let schedule ~cutoff eta k = Some Float.(eta / sqrt (1. + (of_int k / cutoff)))
end

module Make
  (M : T with type Net.input = AD.t and type Lik.data = AD.t) (X : sig
    val eta_sgd_ref : float
    val eta_fl_ref : float
    val eta_sgd : LR.t
    val eta_sgdm : LR.t
    val eta_adam : LR.t
    val eta_fl : LR.t
    val eta_fl_aux : LR.t
    val bs : int
    val bs_curvature : int
    val beta : float
    val weight_decay : float option
    val aux_weight_decay : float option
    val damping : float option
    val n_aux_iter : int -> int
    val save_loss_every : int option
    val save_lambda_every : int option
  end) =
struct
  include X

  let in_dir = Cmdargs.in_dir "-d"
  let n_iter = Cmdargs.(get_int "-n_iter" |> default 50000)

  let info_file, algo =
    match
      Cmdargs.(get_string "-algo" |> force ~usage:"-algo [sgd | sgdm | adam | fl]")
    with
    | "sgd" as s -> s, `sgd
    | "sgdm" as s -> s, `sgdm
    | "adam" as s -> s, `adam
    | "fl" as s -> s, `fl
    | _ -> assert false

  let () =
    if C.first
    then (
      ignore (Bos.Cmd.(v "rm" % "-f" % in_dir info_file) |> Bos.OS.Cmd.run);
      ignore (Bos.Cmd.(v "rm" % "-f" % in_dir (info_file ^ "_log")) |> Bos.OS.Cmd.run))

  let config =
    { learning_rate = eta_fl
    ; learning_rate_aux = eta_fl_aux
    ; bs
    ; bs_curvature
    ; beta
    ; damping
    ; weight_decay
    ; n_aux_iter
    ; adam =
        { Prms.Opt.Adam.default_config with
          epsilon = 1E-4
        ; weight_decay = aux_weight_decay
        }
    }

  (* proper scale for all our lambda initialisers, to recover SGD initially *)
  let scale = Float.(eta_sgd_ref / eta_fl_ref)
  let losses = ref []
  let log = ref []
  let log_tmp = ref []
  let tick = ref 0.
  let walltime = ref 0.

  let monitor ~things_to_log:(log_every, logs) ~tick ~k ~loss ~aux_loss ~g:_ w =
    if Float.is_nan loss then raise Nan_loss;
    walltime := !walltime +. Unix.gettimeofday () -. !tick;
    let l = [| Float.of_int k; !walltime; loss; aux_loss |] in
    losses := l :: !losses;
    Option.iter save_loss_every ~f:(fun e ->
      if C.first && k % e = 0
      then (
        Stdio.printf "[iter %05i] -- loss = %.5f\n%!" k loss;
        !losses
        |> List.rev
        |> Array.of_list
        |> Owl.Mat.of_arrays
        |> Owl.Mat.save_txt ~append:true ~out:(in_dir info_file);
        !log_tmp
        |> List.rev
        |> Array.of_list
        |> Owl.Mat.of_arrays
        |> Owl.Mat.save_txt ~append:true ~out:(in_dir (info_file ^ "_log"));
        losses := [];
        log_tmp := []));
    (* log stuff *)
    if k % log_every = 0
    then (
      let x = List.map logs ~f:(fun g -> g w) |> Array.of_list in
      let x = Array.append l x in
      log := x :: !log;
      log_tmp := x :: !log_tmp);
    (* reset the time counter such that all the above bookkeeping stuff
       doesn't count towards wall clock time *)
    tick := Unix.gettimeofday ()

  let train ?lambda_hook ~things_to_log ~w ~lambda sample_data =
    losses := [];
    log := [];
    log_tmp := [];
    tick := Unix.gettimeofday ();
    let monitor = monitor ~things_to_log ~tick in
    let monitor_fl ~k ~loss ~aux_loss ~g prms =
      let w, lambda = prms in
      monitor ~k ~loss ~aux_loss ~g w;
      Option.iter save_lambda_every ~f:(fun e ->
        if C.first && k % e = 0 then M.L.save_txt ~prefix:(in_dir "lambda") lambda);
      Option.iter lambda_hook ~f:(fun f -> f ~k ~w ~g ~lambda);
      tick := Unix.gettimeofday ()
    in
    let train_fun =
      match algo with
      | `sgd ->
        M.train_sgd
          ~config:{ Prms.Opt.SGD.default_config with weight_decay }
          ~learning_rate:eta_sgd
          ~monitor
          ~bs
      | `sgdm ->
        M.train_sgdm
          ~config:{ Prms.Opt.SGD_momentum.default_config with weight_decay }
          ~learning_rate:eta_sgdm
          ~monitor
          ~bs
      | `adam ->
        (* take Goldfarb et al's best Adam parameters *)
        M.train_adam
          ~config:
            { Prms.Opt.Adam.default_config with
              beta2 = 0.9
            ; epsilon = 1E-4
            ; weight_decay
            }
          ~learning_rate:eta_adam
          ~monitor
          ~bs
      | `fl -> M.train ~config ~lambda ~monitor:monitor_fl
    in
    let w_opt = train_fun ~w ~n_iter sample_data in
    w_opt, !log |> List.rev |> Array.of_list |> Mat.of_arrays
end
