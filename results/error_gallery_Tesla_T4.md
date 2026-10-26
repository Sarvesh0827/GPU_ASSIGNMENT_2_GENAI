# Error gallery — worst class: **negative**

Recall 0.5726, precision 0.6634, 50 misses on the test split.

Label order for the confusion matrix: negative, neutral, positive

Confusion matrix (rows = true, cols = predicted):

```
[[67, 39, 11], [28, 248, 67], [6, 76, 209]]
```

## Five misclassified examples

### 1. true=negative  pred=neutral  (row 4625)

> 2nd attempt at watching PLL Halloween special. I hope my dad doesn't come and ruin it!

Diagnosis: The complaint is understated; without strong sentiment words it scored as news-like neutral.

### 2. true=negative  pred=neutral  (row 1297)

> @user they forget Quran and Islam talked about making them nations to know each other may Allah guide them

Diagnosis: The complaint is understated; without strong sentiment words it scored as news-like neutral.

### 3. true=negative  pred=neutral  (row 574)

> @user That\u2019s what I figured. We suspect you may be experiencing some issues related to the storm\u002c Maria. We\u2019re working on it! ^DGM

Diagnosis: The complaint is understated; without strong sentiment words it scored as news-like neutral.

### 4. true=negative  pred=neutral  (row 141)

> "Watching Les Mis 25th Anniversary concert. How long before the boy Jonas gives me rage, knowing Michael Ball was backstage?"

Diagnosis: The complaint is understated; without strong sentiment words it scored as news-like neutral.

### 5. true=negative  pred=positive  (row 1424)

> Happy Valentine\u2019s Day is trending? ?? But... it\u2019s not even February?

Diagnosis: Positive lexical items (irony/sarcasm) overrode the negative intent.
