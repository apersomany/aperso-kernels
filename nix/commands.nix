{ packages, environment }:

let
  build = builtins.readFile ./commands/build.sh;
  command =
    name:
    packages.writeShellApplication {
      name = "gemm-${name}";
      runtimeInputs = environment.inputs;
      runtimeEnv = environment.variables;
      # Every command builds first; build.sh defines $build_directory
      text = if name == "build" then build else build + builtins.readFile ./commands/${name}.sh;
    };
in
packages.lib.genAttrs [
  "build"
  "run"
  "trace"
  "att"
  "pmc"
] command
