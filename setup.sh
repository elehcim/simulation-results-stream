mkdir -p ~/.streamlit/

echo "[general]
email = \"michele.mastropietro@gmail.com\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml

mkdir -p ~/.config/matplotlib/stylelib
cp stream/MNRAS.mplstyle ~/.config/matplotlib/stylelib/
# conda install numpy

# python -m pip install pynbody
# python -m pip install git+https://github.com/elehcim/simulation.git@master
