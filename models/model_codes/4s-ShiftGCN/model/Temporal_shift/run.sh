cd cuda

rm -rf ./__pycache__
rm -rf ./dist
rm -rf ./build
rm -rf ./shift_cuda_linear_cpp.egg-info

python3 setup.py install --user

cd ..

CUDA_VISIBLE_DEVICES="0" python demo.py
