wget https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AADmGDNsL_6v9BOJDc96PDoza/data-big-test.t7.zip?dl=1
wget https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAD0c3RxA03mpgCL1Ym9sCyGa/data-big-train.t7.zip?dl=1
wget https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AADZDJS_NEbeOgtELT4wyoR3a/data-small-test.t7.zip?dl=1
wget https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AADmMc5Ku4_2j_v1Bhi1ZB-na/data-small-train.t7.zip?dl=1

mv data-big-test.t7.zip?dl=1 data-big-test.t7.zip
mv data-big-train.t7.zip?dl=1 data-big-train.t7.zip
mv data-small-test.t7.zip?dl=1 data-small-test.t7.zip
mv data-small-train.t7.zip?dl=1 data-small-train.t7.zip

unzip data-big-test.t7.zip 
unzip data-big-train.t7.zip
unzip data-small-test.t7.zip
unzip data-small-train.t7.zip 

rm -rf __MACCOSX
rm -rf data-big-test.t7.zip
rm -rf data-big-train.t7.zip  
rm -rf data-small-test.t7.zip
rm -rf data-small-train.t7.zip



