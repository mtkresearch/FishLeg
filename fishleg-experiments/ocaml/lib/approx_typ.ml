open Typ

module type T = sig
  module W : Prms.T
  module L : Prms.T

  val q_v_prod : lambda:L.t' -> W.t' -> W.t'
end

module Diag_low_rank_P = struct
  type ('a, 'b) t =
    { d : 'a
    ; f : 'b
    }
  [@@deriving prms]
end

module Kronecker_P = struct
  type 'a t =
    { i : int
    ; lf : 'a
    ; rf : 'a
    }
  [@@deriving prms]
end

module Block_Kronecker_P = struct
  type ('a, 'b) t =
    { d1 : 'a
    ; d2 : 'a
    ; blocks : 'b
    }
  [@@deriving prms]
end
