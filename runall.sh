
indexes=`eval seq 0 7`
time_transfer=0

for i in ${indexes}
do
./bin/1d_convolution_tests $i $time_transfer
done
