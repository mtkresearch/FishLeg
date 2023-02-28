open Typ

module type T = sig
  module W : Prms.T

  type input
  type output

  (** forward pass *)
  val f : w:W.t' -> input -> output

  val manual_f_df : (tag:int -> w:W.t' -> input -> (output -> AD.t) -> AD.t * W.t') option
end

module MLP_Layer_P = struct
  type 'a t =
    { w : 'a
    ; grad_info : (Mat.mat * Mat.mat) option
    }
  [@@deriving prms]
end
