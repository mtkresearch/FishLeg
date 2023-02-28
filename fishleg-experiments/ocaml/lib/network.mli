open Base
open Typ
include module type of Network_typ

(** Simple one-layer N→1 network *)
module Linear : sig
  include T with type 'a W.p = 'a and type input = AD.t and type output = AD.t

  val init : ?scale:float -> int -> W.t
end

(** Simple MLP with arbitrary layers and activation functions *)
module MLP (Act : sig
  val activation_fun : layer:int -> AD.t -> AD.t
end) : sig
  include
    T
      with module W = Prms.Array(MLP_Layer_P.Make(Prms.P))
       and type input = AD.t
       and type output = AD.t

  val init : ?scale:float -> ?bias:float -> int array -> W.t
end

(** Simple MLP with arbitrary layers and activation functions *)
module MLP_linear_no_bias : sig
  include
    T
      with module W = Prms.Array(MLP_Layer_P.Make(Prms.P))
       and type input = AD.t
       and type output = AD.t

  val init : ?scale:float -> int array -> W.t
end
