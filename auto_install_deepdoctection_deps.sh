set -eu

mkdir -p $HOME/local

POPPLER='poppler-24.04.0'
wget https://poppler.freedesktop.org/$POPPLER.tar.xz
tar -xvf  $POPPLER.tar.xz
cd $POPPLER
./configure --prefix=$HOME/local
make
make install

git clone git@github.com:DanBloomberg/leptonica.git
cd leptonica
git checkout tags/1.83.1
./autogen.sh
./configure --prefix=$HOME/local/
make
make install

grep -i "#define LIBLEPT_MAJOR_VERSION" $HOME/local/include/leptonica/allheaders.h
grep -i "#define LIBLEPT_MINOR_VERSION" $HOME/local/include/leptonica/allheaders.h

git clone git@github.com:tesseract-ocr/tesseract.git
cd tesseract
git checkout tags/5.3.0
./autogen.sh
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig
./configure --prefix=$HOME/local/
make
make install

wget https://github.com/qpdf/qpdf/archive/refs/tags/v11.6.4.tar.gz
tar -xvf v11.6.4.tar.gz
cd qpdf-11.6.4/
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$HOME/local
cmake --build build
cmake --install build
