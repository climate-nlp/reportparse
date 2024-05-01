set -eu

echo "We are installing required packages under $HOME/local. You will need to export the path in shell profile script (e.g., .bashrc)."
export PATH=$HOME/local:$HOME/local/bin:$HOME/local/include:$HOME/local/lib:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH

mkdir -p $HOME/local

# Install poppler 22.12.0
POPPLER='poppler-22.12.0'
wget https://poppler.freedesktop.org/$POPPLER.tar.xz
tar -xvf  $POPPLER.tar.xz
rm $POPPLER.tar.xz
cd $POPPLER
mkdir -p build
cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release -DENABLE_QT6=OFF -DCMAKE_INSTALL_PREFIX=$HOME/local
cmake --build build
cmake --install build

# Install leptonica 1.83.1
git clone https://github.com/DanBloomberg/leptonica
cd leptonica
git checkout tags/1.83.1
./autogen.sh
./configure --prefix=$HOME/local/
make
make install

grep -i "#define LIBLEPT_MAJOR_VERSION" $HOME/local/include/leptonica/allheaders.h
grep -i "#define LIBLEPT_MINOR_VERSION" $HOME/local/include/leptonica/allheaders.h

# Install tesseract 5.3.0
git clone https://github.com/tesseract-ocr/tesseract
cd tesseract
git checkout tags/5.3.0
./autogen.sh
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig
./configure --prefix=$HOME/local/
make
make install

wget https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata --directory-prefix $HOME/local/share/tessdata/

# Install qpdf 11.6.4
QPDF='v11.6.4'
wget https://github.com/qpdf/qpdf/archive/refs/tags/$QPDF.tar.gz
tar -xvf $QPDF.tar.gz
rm $QPDF.tar.gz
cd qpdf-11.6.4/
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$HOME/local
cmake --build build
cmake --install build

