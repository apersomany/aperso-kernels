{
  lib,
  stdenv,
  stdenvNoCC,
  fetchzip,
  makeWrapper,
  patchelf,
  libdrm,
  numactl,
  rdma-core,
}:

let
  target = "gfx120X-all";
  version = "10.0.0";
  compiler_support = "${stdenv.cc}/nix-support";
  libc = stdenv.cc.libc;
  runtime_path = lib.makeLibraryPath [
    stdenv.cc.cc
    libdrm
    numactl
    rdma-core
  ];
in
stdenvNoCC.mkDerivation {
  pname = "therock-rocm-sdk-${target}";
  inherit version;

  src = fetchzip {
    url = "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-${target}-${version}.tar.gz";
    name = "rocm${lib.versions.major version}-${lib.toLower target}-${version}";
    hash = "sha256-c4Cp3fotrJk10xEiJ20dnR4PS3K76sy8/lG6zD81d/A=";
    stripRoot = false;
  };

  nativeBuildInputs = [
    makeWrapper
    patchelf
  ];

  dontUnpack = true;
  dontPatchELF = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall

    cp -a "$src" "$out"
    chmod -R u+w "$out"

    for directory in bin lib lib/host-math/lib lib/llvm/bin lib/llvm/lib lib/rocm_sysdeps/lib share/amd_smi/amdsmi; do
      for file in "$out/$directory"/*; do
        [[ -f $file && ! -L $file ]] && isELF "$file" || continue
        # patchelf corrupts objects whose program headers start beyond the first page
        (( $(od -An -tu8 -j32 -N8 "$file") <= 4096 )) || continue
        patchelf --add-rpath ${runtime_path} "$file"
        if patchelf --print-interpreter "$file" &>/dev/null; then
          patchelf --set-interpreter ${stdenv.cc.bintools.dynamicLinker} "$file"
        fi
      done
    done

    # Clang rejects -mtls-dialect for the amdgcn device target
    makeWrapper "$out/lib/llvm/bin/clang++" "$out/bin/therock-hip-clang++" \
      --add-flags "--gcc-toolchain=${stdenv.cc.cc} --rocm-path=$out" \
      --add-flags "$(sed 's/-mtls-dialect=[^ ]*//' ${compiler_support}/cc-cflags-before)" \
      --add-flags "$(cat ${compiler_support}/{cc-cflags,libc-cflags,libc-crt1-cflags})" \
      --append-flags "-L${libc}/lib -Wl,--dynamic-linker=${stdenv.cc.bintools.dynamicLinker}" \
      --append-flags "-Wl,-rpath,${stdenv.cc.cc.lib}/lib -Wl,-rpath,${libc}/lib" \
      --append-flags "$(cat ${compiler_support}/{libc-ldflags,cc-ldflags})"

    runHook postInstall
  '';

  meta = {
    description = "TheRock ROCm ${version} SDK for ${target}";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
