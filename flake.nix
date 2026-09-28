{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      packages = nixpkgs.legacyPackages.${system};
      rocm = packages.callPackage ./nix/rocm.nix { };
      environment = import ./nix/environment.nix { inherit packages rocm; };
      commands = import ./nix/commands.nix { inherit packages environment; };
      applications = builtins.mapAttrs (name: command: {
        type = "app";
        program = packages.lib.getExe command;
        meta.description = "Run gemm-${name}";
      }) commands;
    in
    {
      apps.${system} = applications // {
        default = applications.run;
      };

      devShells.${system}.default = packages.mkShellNoCC {
        packages = environment.inputs ++ builtins.attrValues commands;
        env = environment.variables;
      };
    };
}
