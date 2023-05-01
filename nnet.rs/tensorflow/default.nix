with import <nixpkgs> {}; {
  sdlEnv = stdenv.mkDerivation {
    name = "sdl";
    shellHook = ''
      export CUDA_TOOLKIT_PATH=${symlinkJoin {
        name = "${cudatoolkit.name}-unsplit";
        paths = [ cudatoolkit.out cudatoolkit.lib ];
      }}

      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib:${symlinkJoin { name = "${cudatoolkit.name}-unsplit"; paths = [ cudatoolkit.out cudatoolkit.lib ];}}/lib:${cudnn}/lib

      export TF_CUDA_VERSION=${cudatoolkit.majorVersion}
      export CUDNN_INSTALL_PATH=${cudnn}
      export TF_CUDNN_VERSION=${cudnn.majorVersion}
      export GCC_HOST_COMPILER_PATH=${cudatoolkit.cc}/bin/gcc

    '';
    nativeBuildInputs = [
      gcc5
    ];
    buildInputs = [
      bazel binutils
      # PYTHON
      python pythonPackages.pip pythonPackages.virtualenv
      # CUDA
      cudatoolkit cudnn linuxPackages.nvidia_x11
    ];
  };
}
