open Base
open Typ
include module type of Approx_typ

(** This module provides a variety of (quadratic) Legendre approximations
    for different types of parameter dictionaries that are used in the models
    of {Network}.
   
    Typically each module will also expose an [init] function that takes a
    [scale] parameter -- this should be set to the ratio [eta_sgd / eta_lf]
    (no sqrt -- this is done internally)

*)

module Full_vector : sig
  (** For simple parameter vectors, we can use a full quadratic form  
    [F(u) = u^T LL^T u] *)

  include T with type 'a W.p = 'a and type 'a L.p = 'a

  val init : scale:float -> int -> L.t
end

module Diag (W : Prms.T) : sig
  (** This is a generic diagonal form that works for any parameter structure. *)

  include T with type 'a W.p = 'a W.p and type 'a L.p = 'a W.p

  val init : scale:float -> W.t -> L.t
end

module Block_Kronecker : sig
  (** Block-Kronecker, specific to MLPs *)

  include
    T
      with module W = Prms.Array(Network.MLP_Layer_P.Make(Prms.P))
       and module L = Block_Kronecker_P.Make
                        (Prms.Option(Prms.Array(Network.MLP_Layer_P.Make(Prms.P))))
                        (Prms.Array(Prms.List(Kronecker_P.Make(Prms.P))))

  val init_diag
    :  ?with_bias:bool
    -> ?with_d1:bool
    -> ?with_d2:bool
    -> scale:float
    -> int array
    -> L.t

  val init_diag2
    :  ?with_bias:bool
    -> ?with_d1:bool
    -> ?with_d2:bool
    -> scale:float
    -> int array
    -> L.t

  val init_tri_diag
    :  ?with_bias:bool
    -> ?with_d1:bool
    -> ?with_d2:bool
    -> scale:float
    -> int array
    -> L.t
end
