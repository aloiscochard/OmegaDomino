with import <nixpkgs> {}; {
  sdlEnv = stdenv.mkDerivation {
    name = "nnet.rs";
      # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib:${symlinkJoin { name = "${cudatoolkit.name}-unsplit"; paths = [ cudatoolkit.out cudatoolkit.lib ];}}/lib:${cudnn}/lib
      # export LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib:${symlinkJoin { name = "${cudatoolkit.name}-unsplit"; paths = [ cudatoolkit.out cudatoolkit.lib ];}}/lib:${cudnn}/lib

    shellHook = ''
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib
      export LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib
      export OPENSSL_INCLUDE_DIR=${openssl.dev}
      export OPENSSL_DIR=${openssl.out}
      export NIX_LABEL="nnet.rs"
      python3 -m venv .env
      source .env/bin/activate
      pip install pipenv
      pip install tensorflow==1.14.0
    '';
    buildInputs = [
      rustup openssl
      python37
      # python37Packages.tensorflow
      # CUDA
      # cudatoolkit cudnn
    ];
  };
}
