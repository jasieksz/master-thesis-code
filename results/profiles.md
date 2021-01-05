## 3C 3V
* all profiles `2^9 = 512`
* VCR ILP detection `506`
* VCR not VR nor CR `0`

## 4C 4V
* all profiles `2^16 = 65536`
* VCR ILP detection `57832`
* VCR not VR nor CR `96`

## 5C 5V
```
+-------+----------+
|    key|sum(value)|
+-------+----------+
|NCOPVCR|    501000|
|    VCR|  19846082|
+-------+----------+
```

## 6C 4V
```
+-------+----------+
|    key|sum(value)|
+-------+----------+
|NCOPVCR|    144480|
|    VCR|  10893736|
+-------+----------+
```

## 4C 6V
```
+-------+----------+
|    key|sum(value)|
+-------+----------+
|NCOPVCR|    144480|
|    VCR|  10893736|
+-------+----------+
```

## Examples 5C 5V
```
Profile(A=
array([[0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0.],
       [0., 0., 1., 1., 0.],
       [0., 1., 0., 1., 0.],
       [1., 0., 0., 1., 0.]]),
    C=[
        Candidate(id='A', x=2.0, r=0.0), 
        Candidate(id='B', x=0.0, r=0.0), 
        Candidate(id='C', x=1.0, r=0.0), 
        Candidate(id='D', x=1.0, r=1.0), 
        Candidate(id='E', x=3.0, r=0.0)],
    V=[
        Voter(id='V0', x=4.0, r=0.0), 
        Voter(id='V1', x=0.0, r=2.0), 
        Voter(id='V2', x=1.0, r=0.0), 
        Voter(id='V3', x=0.0, r=0.0), 
        Voter(id='V4', x=2.0, r=0.0)])
```
```
Profile(A=
array([[0., 0., 1., 0., 1.],
       [0., 0., 1., 1., 0.],
       [1., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 1.]]), 
    C=[
        Candidate(id='A', x=0.0, r=0.0), 
        Candidate(id='B', x=0.0, r=1.0), 
        Candidate(id='C', x=1.0, r=2.0), 
        Candidate(id='D', x=3.0, r=0.0), 
        Candidate(id='E', x=2.0, r=0.0)], 
    V=[
        Voter(id='V0', x=2.0, r=0.0), 
        Voter(id='V1', x=4.0, r=1.0),
        Voter(id='V2', x=0.0, r=0.0), 
        Voter(id='V3', x=4.0, r=0.0), 
        Voter(id='V4', x=2.0, r=1.0)])
```

```
[Profile(A=
array([[1., 1., 0., 0., 1.],
       [1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 1.],
       [0., 0., 1., 0., 1.]]), 
    C=[
        Candidate(id='A', x=3.0, r=1.0), 
        Candidate(id='B', x=2.0, r=0.0), 
        Candidate(id='C', x=0.0, r=0.0), 
        Candidate(id='D', x=1.0, r=0.0), 
        Candidate(id='E', x=1.0, r=1.0)], 
    V=[
        Voter(id='V0', x=2.0, r=0.0), 
        Voter(id='V1', x=0.0, r=2000000000.0), 
        Voter(id='V2', x=3.0, r=0.0), 
        Voter(id='V3', x=1.0, r=0.0), 
        Voter(id='V4', x=0.0, r=0.0)])]

```

```
[Profile(A=
array([[1., 1., 1., 1., 0.],
       [0., 1., 1., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 1., 0.],
       [1., 0., 1., 0., 0.]]), 
    C=[
        Candidate(id='A', x=1.0, r=0.0), 
        Candidate(id='B', x=3.0, r=1.0), 
        Candidate(id='C', x=0.0, r=2.0), 
        Candidate(id='D', x=0.0, r=0.0), 
        Candidate(id='E', x=3.0, r=0.0)], 
    V=[
        Voter(id='V0', x=0.0, r=2.0), 
        Voter(id='V1', x=2.0, r=0.0), 
        Voter(id='V2', x=4.0, r=0.0), 
        Voter(id='V3', x=0.0, r=0.0), 
        Voter(id='V4', x=1.0, r=0.0)])]
```