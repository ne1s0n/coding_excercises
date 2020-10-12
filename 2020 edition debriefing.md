Things we (Nelson and Filippo) would improve:

* more focus on learning rate, show the effect of too big/too small
	* since we are at it, give [learning rate autofinder](https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/ ) a go 
* show practical examples of bad cases:
    * optimizer is not working, loss keeps growing (what happens changing)
    * actual overfitting, effect of regularization
* have a single dataset that should be used along the whole course (or a couple: an easy one and a difficult one): probably at least a couple: one for demonstrations by the teachers, the other for interactive/collaborative exercises with the students (maybe aditional ones if we want to show other types of problems: e.g. multiclass classification, regression, pattern recognition ...)
* consider these topics:
    * batch normalization
    * learning rate decay
    * 1D convolution for time-series and genotypes
    * p > n problems: how to do this with DL; is there an advantage over other ML methods? Or is DL better at making use of problems where n is very large?
* keep the drawing with the pen agile. One student told that sometimes it took too much time
* "finetune" the pace of lectures (sometimes too quick, especially for me -Filippo- in the first days)
* more guided/shared exercises. The teacher writes, the student tell what to do (give it a shot to volunteering students to lead the writing? if any ...)
* a fifth day? (probably needed if we plan to add material)
* simplify the first demonstration of a DL model for image recognition: it seemed a bit overwhelming for some of the students, some details can be skipped and groupd into the basic steps that we outline at the beginning (set up, data management, model building, training, evauation)

General TODO:

* complete prep exercises on normalization (and maybe others)
* cleanup the repo so that it does not explode in size due to binary blobs
