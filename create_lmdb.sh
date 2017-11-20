MY=my-projects/re_test

echo "Create train lmdb.."
rm -rf $MY/train_lmdb
build/tools/convert_imageset --shuffle --resize_height=256 --resize_width=256 /home/walter040422/Documents/ $MY/train.txt $MY/train_lmdb



echo "Create val lmdb.."
rm -rf $MY/val_lmdb
build/tools/convert_imageset --shuffle --resize_height=256 --resize_width=256 /home/walter040422/Documents/ $MY/val.txt $MY/val_lmdb


echo "All Done.."
