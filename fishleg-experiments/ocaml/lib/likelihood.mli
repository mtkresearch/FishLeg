open Base
open Typ
include module type of Likelihood_typ

(** Simple Gaussian likelihood for array-typed data, with uniform variance *)
module Gaussian : sig
  include
    T
      with type stats = AD.t
       and type data = AD.t
       and type 'a W.p = 'a
       and type 'a L.p = 'a

  val init : float -> W.t
  val init_lambda : scale:float -> L.t
  val mean_squared_error : obs:AD.t -> AD.t -> float
end

(** Bernoulli *)
module Bernoulli : sig
  include
    T
      with type stats = AD.t (* logits *)
       and type data = AD.t
       and module W = Prms.None
       and module L = Prms.None

  val binary_cross_entropy : obs:AD.t -> AD.t -> float
  val mean_squared_error : obs:AD.t -> AD.t -> float
end

(** Continuous Bernoulli *)
module Continuous_Bernoulli : sig
  include
    T
      with type stats = AD.t (* logits *)
       and type data = AD.t
       and module W = Prms.None
       and module L = Prms.None

  val binary_cross_entropy : obs:AD.t -> AD.t -> float
  val mean_squared_error : obs:AD.t -> AD.t -> float
end

(** Softmax (categorical) likelihood, with one-hot data encoding *)
module Softmax : sig
  include
    T
      with type stats = AD.t
      (* log-probabilities, unnormalised *)
       and type data = AD.t
      (* one-hot vectors *)
       and module W = Prms.None
       and module L = Prms.None

  (** in addition, we expose the % accuracy function *)
  val accuracy : AD.t -> AD.t -> float
end
