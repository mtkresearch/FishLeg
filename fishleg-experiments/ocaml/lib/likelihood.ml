open Base
module U = Utils (* shadowed by Owl, grr *)
open Owl
open Typ
include Likelihood_typ

let mse ~obs:y mu =
  let z = Float.(1. / of_int (AD.numel y)) in
  AD.Maths.(F z * sum' (sqr (y - mu))) |> AD.unpack_flt

(* ----------------------------------------------------------- *)

module Gaussian = struct
  module W = Prms.P
  module L = Prms.P

  type stats = AD.t
  type data = AD.t

  let init x = Prms.free (AD.F x)
  let init_lambda ~scale = Prms.free (AD.F (Float.sqrt scale))
  let sample ~w:sigma mu = AD.Maths.(mu + (AD.Arr.(gaussian (shape mu)) * sigma))

  let neg_logp ~w:sigma ~obs:y mu =
    let logsigma2 = AD.Maths.(log (sqr sigma)) in
    (* let z = Float.(0.5 / of_int (AD.numel y)) in *)
    let z = Float.(0.5 / of_int (AD.row_num y)) in
    AD.Maths.(F z * sum' (logsigma2 + sqr ((y - mu) / sigma)))

  (* analytical expression below doesn't seem to be making a difference *)
  let expected_neg_logp = None
  (* Some
      (fun ~expectation_under:(mu, sigma) (mu', sigma') ->
        let d = Float.of_int (AD.numel mu) in
        (* let d = Float.of_int (AD.row_num mu) in *)
        let z = Float.log Const.pi2 in
        AD.Maths.(
          F 0.5
          / F d
          * ((F d * (F z + sqr (sigma / sigma') + log (sqr sigma')))
            + l2norm_sqr' ((mu - mu') / sigma')))) *)

  let q_v_prod ~lambda u = AD.Maths.(sqr lambda * u)
  let mean_squared_error = mse
end

(* ----------------------------------------------------------- *)

(* NOTE: this likelihood assumes that the parameter is pre-sigmoid *)
module Bernoulli = struct
  module W = Prms.None
  module L = Prms.None

  type stats = AD.t
  type data = AD.t

  (* sampling from Bernoulli(p) where p = 1/(1+exp(-mu)) is the same as
     drawing a logistic rv with scale=1 and loc=mu and comparing to zero;
     tested: OK  *)
  let sample ~w:_ mu =
    Arr.map
      (fun mu -> if Float.(Stats.logistic_rvs ~scale:1. ~loc:mu < 0.) then 0. else 1.)
      (AD.unpack_arr mu)
    |> AD.pack_arr

  (* binary entropy loss  = negative log likelihood up to a constant *)
  let neg_logp ~w:_ ~obs:y mu =
    let n = Mat.row_num (AD.unpack_arr (AD.primal' mu)) in
    let bce = AD.Maths.(sum' ((mu * (F 1. - y)) + U.softplus (neg mu))) in
    AD.Maths.(bce / F Float.(of_int n))

  (* binary entropy loss = negative log likelihood up to a constant *)
  let binary_cross_entropy ~obs mu = AD.unpack_flt (neg_logp ~w:() ~obs mu)
  let mean_squared_error ~obs y = mse ~obs (AD.Maths.sigmoid y)
  let expected_neg_logp = None
  let q_v_prod ~lambda:_ _ = ()
end

(* ----------------------------------------------------------- *)

(* NOTE: this likelihood assumes that the parameter is pre-sigmoid *)
module Continuous_Bernoulli = struct
  module W = Prms.None
  module L = Prms.None

  type stats = AD.t
  type data = AD.t

  let reg z =
    let eps = AD.F 1E-8 in
    AD.Maths.(U.sign_not_zero z * (eps + abs z))

  let change_var mu =
    (* z = (1 - exp(-mu)) / (1 + exp (-mu))  *)
    let enmu = AD.Maths.(exp (neg mu)) in
    AD.Maths.((F 1. - enmu) / (F 1. + enmu)) |> reg

  let inv_cdf mu y =
    let z = change_var mu in
    let l1pz = U.log1p z in
    let l1mz = U.log1p AD.Maths.(neg z) in
    AD.Maths.((U.log1p (((F 2. * y) - F 1.) * z) - l1mz) / (l1pz - l1mz))

  (* pass uniform samples through the inverse CDF *)
  let sample ~w:_ mu = AD.Arr.uniform ~a:0. ~b:1. (AD.shape mu) |> inv_cdf mu

  (* binary entropy loss = negative log likelihood up to a constant *)
  let binary_cross_entropy ~obs:y mu =
    let n = AD.Mat.row_num mu in
    let bce = AD.Maths.(sum' ((mu * (F 1. - y)) + U.softplus (neg mu))) in
    Float.(AD.unpack_flt bce / of_int n)

  let mean_squared_error = Bernoulli.mean_squared_error

  (* binary entropy loss  = negative log likelihood up to a constant *)
  let neg_logp ~w:_ ~obs:y mu =
    let n = AD.Mat.row_num mu in
    (* add the log partition function *)
    let log_partition =
      let z = change_var mu in
      (* we exploit the fact that log( (1+z)/(1-z) ) / z  is an even function of z,
         so we can compute it for |z| and take the log of it directly... *)
      let az = AD.Maths.abs z in
      AD.Maths.(log (U.log1p az - U.log1p (neg az)) - log az)
    in
    AD.Maths.(
      sum' (log_partition + (mu * (F 1. - y)) + U.softplus (neg mu)) / F Float.(of_int n))

  let expected_neg_logp = None
  let q_v_prod ~lambda:_ _ = ()
end

(* ----------------------------------------------------------- *)

module Softmax = struct
  (* this one doesn't introduce any parameters *)
  module W = Prms.None
  module L = Prms.None

  type stats = AD.t
  type data = AD.t

  let sample ~w:() logp =
    (* this is logp up to a constant -- doesn't matter  *)
    let logp = AD.unpack_arr logp in
    (* Gumbel max trick *)
    let g = Arr.(uniform ~a:0. ~b:1. (shape logp)) in
    let z = Arr.(logp - log (neg (log g))) in
    let zmax =
      Mat.map_rows
        (fun row ->
          let _, i = Mat.max_i row in
          i.(1))
        z
    in
    let y = Arr.(zeros (shape logp)) in
    Array.iteri zmax ~f:(fun i j -> Mat.set y i j 1.);
    AD.Arr y

  let neg_logp ~w:() ~obs:y logp =
    let p = AD.Maths.softmax ~axis:1 logp in
    AD.Maths.(cross_entropy y p / F Float.(of_int (AD.row_num p)))

  let expected_neg_logp = None
  let q_v_prod ~lambda:_ _ = ()

  let accuracy logp y =
    let logp = AD.unpack_arr logp in
    let y = AD.unpack_arr y in
    let a = ref 0 in
    let n = Mat.row_num logp in
    for i = 0 to n - 1 do
      let _, id1 = Mat.max_i (Mat.row logp i) in
      let _, id2 = Mat.max_i (Mat.row y i) in
      if id1.(1) = id2.(1) then Int.incr a
    done;
    Float.(of_int !a / of_int n)
end
