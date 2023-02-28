(* parallelisation of hyperparameter grid search *)
include Comm.Mpi (struct
  let init_rng seed =
    Owl_stats_prng.init seed;
    Random.init (Owl_stats_prng.rand_int ())
end)

let _ =
  Owl_stats_prng.init Cmdargs.(get_int "-seed" |> default 2022);
  Random.init (Owl_stats_prng.rand_int ())
  (* self_init_rng () *)
