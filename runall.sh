
indexes=`eval seq 0 6`
for i in ${indexes}
do
./bin/1d_convolution_tests $i
done
