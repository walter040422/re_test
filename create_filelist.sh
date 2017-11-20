DATA=/home/walter040422/Documents
MY=my-projects/re_test

echo "Create train.txt..."
rm -rf $MY/train.txt
find $DATA/re/train/bus    -name *.jpg | cut -d '/' -f8 | set "s:^:re/train/bus/:"      | sed "s/$/ bus/"     >>$MY/train.txt
find $DATA/re/train/dragon -name *.jpg | cut -d '/' -f8 | set "s:^:re/train/dragon/:" | sed "s/$/ dragon/">>$MY/train.txt
find $DATA/re/train/elephant    -name *.jpg | cut -d '/' -f8 | set "s:^:re/train/elephant/:"      | sed "s/$/ elephant/"     >>$MY/train.txt
find $DATA/re/train/flower    -name *.jpg | cut -d '/' -f8 | set "s:^:re/train/flower/:"      | sed "s/$/ flower/"     >>$MY/train.txt
find $DATA/re/train/horse    -name *.jpg | cut -d '/' -f8 | set "s:^:re/train/horse/:"      | sed "s/$/ horse/"     >>$MY/train.txt


echo "Create val.txt..."
rm -rf $MY/val.txt
find $DATA/re/test/bus -name *.jpg | cut -d '/' -f8 | set "s:^:re/test/bus/:" | sed "s/$/ bus/" >>$MY/val.txt
find $DATA/re/test/dragon -name *.jpg | cut -d '/' -f8 | set "s:^:re/test/dragon/:" | sed "s/$/ dragon/" >>$MY/val.txt
find $DATA/re/test/elephant -name *.jpg | cut -d '/' -f8 | set "s:^:re/test/elephant/:" | sed "s/$/ elephant/" >>$MY/val.txt
find $DATA/re/test/flower -name *.jpg | cut -d '/' -f8 | set "s:^:re/test/flower/:" | sed "s/$/ flower/" >>$MY/val.txt
find $DATA/re/test/horse -name *.jpg | cut -d '/' -f8 | set "s:^:re/test/horse/:" | sed "s/$/ horse/" >>$MY/val.txt

echo "All done"
