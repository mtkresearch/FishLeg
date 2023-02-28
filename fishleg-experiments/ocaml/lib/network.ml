open Base
open Typ
include Network_typ

(* ---------------------------------------------------------------------  *)

module Linear = struct
  module W = Prms.P

  type input = AD.t
  type output = AD.t

  let init ?(scale = 1.) n =
    let s = scale in
    let z = Float.(s / sqrt (of_int n)) in
    let w = AD.pack_arr Mat.(z $* gaussian n 1) in
    Prms.free w

  let f ~w x = AD.Maths.(x *@ w)
  let manual_f_df = None
end

(* ---------------------------------------------------------------------  *)

module MLP (Act : sig
  val activation_fun : layer:int -> AD.t -> AD.t
end) =
struct
  module W = Prms.Array (MLP_Layer_P.Make (Prms.P))
  open MLP_Layer_P

  type input = AD.t
  type output = AD.t

  let f ~w x =
    Array.foldi w ~init:x ~f:(fun i accu z ->
        let b = AD.Maths.get_slice [ [ 0 ] ] z.w in
        let w = AD.Maths.get_slice [ [ 1; -1 ] ] z.w in
        AD.Maths.(Act.activation_fun ~layer:i (b + (accu *@ w))))

  (* tested against the gradient of [f] above: OK *)
  let manual_f_df =
    let mfdf ~tag ~w x =
      let x = AD.make_reverse x tag in
      let y_pred, acts =
        Array.foldi w ~init:(x, []) ~f:(fun i (accu, info) z ->
            let prev_layer_act =
              let z = AD.Mat.ones (AD.row_num accu) 1 in
              AD.Maths.(concat ~axis:1 z accu)
            in
            let layer_potential = AD.Maths.(prev_layer_act *@ z.w) in
            let accu = Act.activation_fun ~layer:i layer_potential in
            accu, (prev_layer_act, layer_potential) :: info)
      in
      let info = Array.of_list (List.rev acts) in
      fun loss_fun ->
        let loss = loss_fun y_pred in
        AD.reverse_prop (F 1.) loss;
        ( AD.primal' loss
        , Array.map info ~f:(fun (h, e) ->
              let h = AD.unpack_arr (AD.primal h) in
              let e = AD.unpack_arr (AD.adjval e) in
              let ht = Mat.transpose h in
              let g = Mat.(ht *@ e) in
              let n, bs = Mat.shape ht in
              let m = Mat.col_num e in
              (* it only makes sense to exploit the low-rank structure
               if the batch size is small; otherwise it's more efficient
               to perform full matrix-multiplications *)
              if bs < n || bs < m
              then { w = AD.Arr g; grad_info = Some (ht, e) }
              else { w = AD.Arr g; grad_info = None }) )
    in
    Some mfdf

  let init ?(scale = 1.0) ?(bias = 0.) sizes =
    let scal = scale in
    Array.init
      (Array.length sizes - 1)
      ~f:(fun i ->
        let n = sizes.(i)
        and m = sizes.(i + 1) in
        let b = AD.Arr Mat.(bias $* gaussian 1 m) in
        let w = AD.Maths.(F Float.(scal / sqrt (of_int n)) * AD.Mat.gaussian n m) in
        (* result has size sizes.(i) + 1, sizes.(i+1) *)
        { w = Prms.free (AD.Maths.concatenate ~axis:0 [| b; w |]); grad_info = None })
end

(* ---------------------------------------------------------------------  *)

(* special case of deep linear network without biases *)
module MLP_linear_no_bias = struct
  module W = Prms.Array (MLP_Layer_P.Make (Prms.P))
  open MLP_Layer_P

  type input = AD.t
  type output = AD.t

  let f ~w x = Array.fold w ~init:x ~f:(fun accu z -> AD.Maths.(accu *@ z.w))

  (* tested against the gradient of [f] above: OK *)
  let manual_f_df =
    let mfdf ~tag ~w x =
      let x = AD.make_reverse x tag in
      let y_pred, acts =
        Array.fold w ~init:(x, []) ~f:(fun (accu, info) z ->
            let prev_layer_act = accu in
            let accu = AD.Maths.(prev_layer_act *@ z.w) in
            accu, (prev_layer_act, accu) :: info)
      in
      let info = Array.of_list (List.rev acts) in
      fun loss_fun ->
        let loss = loss_fun y_pred in
        AD.reverse_prop (F 1.) loss;
        ( AD.primal' loss
        , Array.map info ~f:(fun (h, e) ->
              let h = AD.unpack_arr (AD.primal h) in
              let e = AD.unpack_arr (AD.adjval e) in
              let ht = Mat.transpose h in
              let g = Mat.(ht *@ e) in
              let n, bs = Mat.shape ht in
              let m = Mat.col_num e in
              (* it only makes sense to exploit the low-rank structure
               if the batch size is small; otherwise it's more efficient
               to perform full matrix-multiplications *)
              if bs < n || bs < m
              then { w = AD.Arr g; grad_info = Some (ht, e) }
              else { w = AD.Arr g; grad_info = None }) )
    in
    Some mfdf

  let init ?(scale = 1.0) sizes =
    let scal = scale in
    Array.init
      (Array.length sizes - 1)
      ~f:(fun i ->
        let n = sizes.(i)
        and m = sizes.(i + 1) in
        let w = AD.Maths.(F Float.(scal / sqrt (of_int n)) * AD.Mat.gaussian n m) in
        { w = Prms.free w; grad_info = None })
end
