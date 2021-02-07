# OID to coco converter
![alt text](https://warehouse-camo.ingress.cmh1.psfhosted.org/6a042d910c74fbce532a01da853019c164ef42a8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e362b2d626c75652e737667 (https://www.python.org/))
python script converts open image dataset to coco format.


### Plugins

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| parameter | meaning |
| ------ | ------ |
| a | path to annotation file |
| i | path to image sizes file |
| d | path to class description file |
| r | path to rotation file |
| c | class name (if set chunk size will be ignored) |
| s | chunk size in each json |


## Usage
**sinle class "Aircraft"**  
`python3 tococo.py --a "train-annotations.csv" --i "train-sizes.csv" --d "class-descriptions.csv" --r "train-rotation.csv" --c "Aircraft"`

**complete OID with specifying chunk size "**  
`python3 tococo.py --a "train-annotations.csv" --i "train-sizes.csv" --d "class-descriptions.csv" --r "train-rotation.csv" --s 25`

**complete OID (Default chunk 19 class) "**  
`python3 tococo.py --a "train-annotations.csv" --i "train-sizes.csv" --d "class-descriptions.csv" --r "train-rotation.csv" `
## License
[MIT](https://choosealicense.com/licenses/mit/)
