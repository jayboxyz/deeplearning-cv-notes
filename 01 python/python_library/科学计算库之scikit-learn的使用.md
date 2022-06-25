# scikit-learn 学习

## 1. sklearn.preprocessing.LabelEncoder

sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内

以数字标签为例：

``` xml
In [1]: from sklearn import preprocessing
   ...: le = preprocessing.LabelEncoder()
   ...: le.fit([1,2,2,6,3])
   ...:
Out[1]: LabelEncoder()
```

参考：[sklearn.preprocessing.LabelEncoder](<https://blog.csdn.net/kancy110/article/details/75043202>)

LabelEncoder可以将标签分配一个0—n_classes-1之间的编码。将各种标签分配一个可数的连续编号：

``` python
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6]) # Transform Categories Into Integers
array([0, 0, 1, 2], dtype=int64)
>>> le.inverse_transform([0, 0, 1, 2]) # Transform Integers Into Categories
array([1, 1, 2, 6])
```



``` python
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) # Transform Categories Into Integers
array([2, 2, 1], dtype=int64)
>>> list(le.inverse_transform([2, 2, 1])) #Transform Integers Into Categories
['tokyo', 'tokyo', 'paris']
```

——from：[使用sklearn之LabelEncoder将Label标准化](<https://blog.csdn.net/u010412858/article/details/78386407>)



## 2. 



## 3. 



