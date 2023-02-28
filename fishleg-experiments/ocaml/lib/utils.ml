open Base
open Typ

let timeit ~label f x =
  let t0 = Unix.gettimeofday () in
  let result = f x in
  let t1 = Unix.gettimeofday () in
  Stdio.printf "[%s] %f\n%!" label Float.(t1 - t0);
  result

let print_dim ~label x =
  let dims = Array.map (AD.shape x) ~f:Int.to_string |> Array.to_list in
  Stdio.printf "[%s] %s" label (String.concat ~sep:" x " dims)

(* define the Heaviside function as in jax *)
let rec heaviside ~at_zero = Stdlib.Lazy.force _heaviside ~at_zero

and ___h ~at_zero x =
  if Float.(x < 0.) then 0. else if Float.(x > 0.) then 1. else at_zero

and _heaviside =
  let open AD.Builder in
  lazy
    (fun ~at_zero ->
      let ___h = ___h ~at_zero in
      build_siso
        (module struct
          let label = "heaviside"
          let ff_f x = AD.F (___h x)
          let ff_arr x = AD.pack_arr (Arr.map ___h x)

          let df cp _ _ =
            match AD.primal' cp with
            | F _ -> AD.F 0.
            | Arr z -> Arr.(zeros (shape z)) |> AD.pack_arr
            | _ -> assert false

          let dr x _ _ =
            match AD.primal' x with
            | F _ -> AD.F 0.
            | Arr z -> Arr.(zeros (shape z)) |> AD.pack_arr
            | _ -> assert false
        end : AD.Builder.Siso))

let sign_not_zero x = AD.Maths.(heaviside ~at_zero:1. x - heaviside ~at_zero:0. (neg x))

(* define the Heaviside function as in jax *)
let rec log1p x = Stdlib.Lazy.force _log1p x
and ___f = Base.Float.log1p

and _log1p =
  let open AD.Builder in
  lazy
    (build_siso
       (module struct
         let label = "heaviside"
         let ff_f x = AD.F (___f x)
         let ff_arr x = AD.pack_arr (Arr.map ___f x)
         let df _cp ap at = AD.Maths.(at / (F 1. + ap))
         let dr ap _cp ca = AD.Maths.(!ca / (F 1. + ap))
       end : AD.Builder.Siso))

let softplus x =
  let c = Arr.max' (AD.unpack_arr (AD.primal' x)) |> AD.pack_elt in
  AD.Maths.(c + log (exp (neg c) + exp (x - c)))
