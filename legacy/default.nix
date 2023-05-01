with import <nixpkgs> {}; {
  sdlEnv = stdenv.mkDerivation {
    name = "anna";
      # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib:${symlinkJoin { name = "${cudatoolkit.name}-unsplit"; paths = [ cudatoolkit.out cudatoolkit.lib ];}}/lib:${cudnn}/lib
      # export LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib:${symlinkJoin { name = "${cudatoolkit.name}-unsplit"; paths = [ cudatoolkit.out cudatoolkit.lib ];}}/lib:${cudnn}/lib
      # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib
      # export LIBRARY_PATH=$LD_LIBRARY_PATH:${libtensorflow}:${stdenv.cc.cc.lib}/lib

    shellHook = ''
      export NIX_LABEL="anna"
      export RUST_BACKTRACE=1
      export RUST_LOG="anna=debug"

      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib
      export LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib

      export OPENSSL_INCLUDE_DIR=${openssl.dev}
      export OPENSSL_DIR=${openssl.out}
      python3 -m venv .env
      source .env/bin/activate
      pip install pipenv
      pip install tensorflow==1.14.0
    '';
    buildInputs = [
      rustup
      pkgconfig
      xorg.libX11 xorg.libXrandr
      sqlite
      stdenv.cc.cc.lib

      # Profiling
      gperftools

      # NNet/TensorFlow
      openssl
      python37

      # CUDA
      # cudatoolkit cudnn

      # player-cli
      ncurses

      # vision

      # utils
      xorg.xwininfo
    ];
  };
}
