sudo apt-get update

# Install libraries
sudo apt-get install -y build-essential libblas-dev liblapack-dev libssl-dev g++ python-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev checkinstall libreadline-dev
sudo apt-get install -y libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev wget curl llvm libncurses5-dev xz-utils tk-dev liblzma-dev python-openssl libreadline-dev unixodbc-dev
sudo apt-get install -y python-setuptools ninja-build

cd ~
mkdir temp
cd temp

# Install or upgrade CMake
sudo apt-get remove cmake cmake-data
wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
tar -zxvf cmake-3.20.2.tar.gz
cd cmake-3.20.2
./bootstrap
make 
sudo make install
echo "export PATH=/usr/local/share/cmake-3.20:\$PATH" >> ~/.bashrc

# Install Boost
wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz 
tar -xf boost_1_76_0.tar.gz
rm boost_1_76_0.tar.gz
cd boost_1_76_0
./bootstrap.sh
./b2
sudo ./b2 install
rm -rf boost_1_76_0

# Install Blaze
wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz
tar -xvf blaze-3.8.tar.gz
rm -rf blaze-3.8.tar.gz
cd blaze-3.8/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
sudo make install
cd ..
rm -rf blaze-3.8/

# Install Pyenv and Python 3.8
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.8.8

