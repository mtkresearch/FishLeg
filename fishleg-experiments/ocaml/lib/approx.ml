open Base
open Typ
include Approx_typ

(* -------------------------------------------------------------------------- *)

module Full_vector = struct
  module W = Prms.P
  module L = Prms.P

  let init ~scale n = AD.pack_arr Mat.(Float.sqrt scale $* eye n) |> Prms.free
  let q_v_prod ~lambda u = AD.Maths.(transpose lambda *@ (lambda *@ u))
end

(* -------------------------------------------------------------------------- *)

module Diag (W : Prms.T) = struct
  module W = W
  module L = W

  let init ~scale:s w =
    let w = W.value w in
    W.(F Float.(sqrt s) $* ones_of w) |> L.map ~f:Prms.free

  let q_v_prod ~lambda u = W.map2 lambda u ~f:(fun d u -> AD.Maths.(sqr d * u))
end

(* -------------------------------------------------------------------------- *)

let n_and_m ~with_bias ~sizes i =
  let n = if with_bias then sizes.(i) + 1 else sizes.(i)
  and m = sizes.(i + 1) in
  n, m

module Block_Kronecker = struct
  open Kronecker_P
  module W = Prms.Array (Network.MLP_Layer_P.Make (Prms.P))

  module L =
    Block_Kronecker_P.Make
      (Prms.Option (W)) (Prms.Array (Prms.List (Kronecker_P.Make (Prms.P))))

  let prod3 a b c =
    let n = AD.Mat.row_num a in
    let m = AD.Mat.row_num c in
    let cost x y = (x * x * y) + (y ** 3) in
    if cost n m < cost m n
    then (
      let tmp = AD.Maths.(a *@ b) in
      AD.Maths.(tmp *@ c))
    else (
      let tmp = AD.Maths.(b *@ c) in
      AD.Maths.(a *@ tmp))

  type product =
    | Full of AD.t
    | Low_rank of AD.t * AD.t

  let q_v_prod ~lambda (u : W.t') : W.t' =
    let open Block_Kronecker_P in
    let open Network.MLP_Layer_P in
    let u =
      match lambda.d1 with
      | Some d -> W.(u * d * d)
      | None -> u
    in
    let block_part =
      (* extract triangular parts for diagonal block components *)
      let lambda =
        Array.mapi lambda.blocks ~f:(fun i ->
          List.map ~f:(fun z ->
            if z.i = i
            then { z with lf = AD.Maths.triu z.lf; rf = AD.Maths.tril z.rf }
            else z))
      in
      (* compute the expression in square brackets once and for all
       (and keep it as a list of low-rank factors if appropriate) *)
      let tmp =
        Array.map
          lambda
          ~f:
            (List.fold ~init:[] ~f:(fun accu z ->
               let g = u.(z.i) in
               let tmp =
                 match g.grad_info with
                 | None -> Full (prod3 z.lf g.w z.rf)
                 | Some (ht, e) ->
                   (* we can maintain the factored representation *)
                   Low_rank (AD.Maths.(z.lf *@ Arr ht), AD.Maths.(Arr e *@ z.rf))
               in
               match accu, tmp with
               (* make sure Full is always the first thing in the list *)
               | Full a :: rest, Full b -> Full AD.Maths.(a + b) :: rest
               | _, Full b -> Full b :: accu
               | _, _ -> accu @ [ tmp ]))
      in
      Array.mapi lambda ~f:(fun i layers ->
        let d =
          List.fold layers ~init:None ~f:(fun accu z ->
            let lft = if i = z.i then AD.Maths.transpose z.lf else z.lf in
            let rft = if i = z.i then AD.Maths.transpose z.rf else z.rf in
            List.fold tmp.(z.i) ~init:accu ~f:(fun accu' p ->
              let tmp =
                match p with
                | Full a -> prod3 lft a rft
                | Low_rank (l, r) ->
                  let l = AD.Maths.(lft *@ l) in
                  let r = AD.Maths.(r *@ rft) in
                  AD.Maths.(l *@ r)
              in
              match accu' with
              | None -> Some tmp
              | Some a -> Some AD.Maths.(a + tmp)))
        in
        Network.MLP_Layer_P.{ w = Option.value_exn d; grad_info = None })
    in
    match lambda.d2 with
    | Some d -> W.map2 d block_part ~f:(fun d b -> AD.Maths.(sqr d * b))
    | None -> block_part

  let make_d ~with_bias ~sizes =
    let _L = Array.length sizes - 1 in
    Array.init _L ~f:(fun i ->
      let n, m = n_and_m ~with_bias ~sizes i in
      Network.MLP_Layer_P.{ w = AD.Mat.ones n m; grad_info = None })

  let init_diag ?(with_bias = true) ?(with_d1 = false) ?(with_d2 = false) ~scale:s sizes
    : L.t
    =
    let factor = Float.(sqrt s) in
    let _L = Array.length sizes - 1 in
    let d1 = if with_d1 then Some (make_d ~with_bias ~sizes) else None in
    let d2 = if with_d2 then Some (make_d ~with_bias ~sizes) else None in
    let blocks =
      Array.init _L ~f:(fun i ->
        let n, m = n_and_m ~with_bias ~sizes i in
        let lf = AD.pack_arr Mat.(Float.(sqrt factor) $* eye n) in
        let rf = AD.pack_arr Mat.(Float.(sqrt factor) $* eye m) in
        [ { i; lf; rf } ])
    in
    Block_Kronecker_P.{ d1; d2; blocks } |> L.map ~f:Prms.free

  let init_diag2 ?(with_bias = true) ?(with_d1 = false) ?(with_d2 = false) ~scale:s sizes
    : L.t
    =
    let factor = Float.(sqrt s) in
    let _L = Array.length sizes - 1 in
    let d1 = if with_d1 then Some (make_d ~with_bias ~sizes) else None in
    let d2 = if with_d2 then Some (make_d ~with_bias ~sizes) else None in
    let blocks =
      Array.init _L ~f:(fun i ->
        let n, m = n_and_m ~with_bias ~sizes i in
        let lf1 = AD.pack_arr Mat.(Float.(sqrt factor) $* eye n) in
        let rf1 = AD.pack_arr Mat.(Float.(sqrt factor) $* eye m) in
        let lf2 =
          AD.pack_arr Mat.(Float.(0.1 * sqrt factor / sqrt (of_int n)) $* gaussian n n)
        in
        let rf2 =
          AD.pack_arr Mat.(Float.(0.1 * sqrt factor / sqrt (of_int m)) $* gaussian m m)
        in
        [ { i; lf = lf1; rf = rf1 }; { i; lf = lf2; rf = rf2 } ])
    in
    Block_Kronecker_P.{ d1; d2; blocks } |> L.map ~f:Prms.free

  let init_tri_diag
    ?(with_bias = true)
    ?(with_d1 = false)
    ?(with_d2 = false)
    ~scale:s
    sizes
    : L.t
    =
    let factor = Float.(sqrt s) in
    let _L = Array.length sizes - 1 in
    let d1 = if with_d1 then Some (make_d ~with_bias ~sizes) else None in
    let d2 = if with_d2 then Some (make_d ~with_bias ~sizes) else None in
    let blocks =
      Array.init _L ~f:(fun i ->
        let n, m = n_and_m ~with_bias ~sizes i in
        let lf = AD.pack_arr Mat.(Float.(sqrt factor) $* eye n) in
        let rf = AD.pack_arr Mat.(Float.(sqrt factor) $* eye m) in
        let z = [ { i; lf; rf } ] in
        (* try to add off-diag block *)
        if i = _L - 1
        then z
        else (
          let i' = i + 1 in
          let n', m' = n_and_m ~with_bias ~sizes i' in
          { i = i'
          ; lf = AD.pack_arr Mat.(gaussian ~sigma:0.01 n n' /$ Float.(of_int n))
          ; rf = AD.pack_arr Mat.(gaussian ~sigma:0.01 m' m /$ Float.(of_int m))
          }
          :: z))
    in
    Block_Kronecker_P.{ d1; d2; blocks } |> L.map ~f:Prms.free
end
