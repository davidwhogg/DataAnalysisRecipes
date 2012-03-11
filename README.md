# Data Analysis Recipes

Chapters from Hogg's non-existent book.

### Authors: ###

(Contributions have come from all of the following.)

* David W. Hogg, New York University
* Jo Bovy, Institute for Advanced Study
* Dustin Lang, Princeton University

### License: ###

**Copyright 2010 2011 2012 the authors.  All rights reserved.**

If you have interest in using or re-using any of this content, get in
touch with Hogg.

### Notes to self: ###

When I want to import stuff from the old SVN repository, I do the
following:

1. I create a new github repository called `foo` and follow the svn
   import instructions.

2. I `git clone` that repository and do things like move the files into
   a directory structure that won't conflict with the current
   structure, like:
   ```
   cd
   git clone git@github.com:davidwhogg/foo.git
   cd foo
   mkdir straightline
   git mv *.pdf straightline
   # etc
   # . . .
   git commit -a -m "fixed up directory structure"
   git push
   ```

3. I make a subtree merge or something like that (I am new to all this)
   like so:
   ```
   cd
   cd DataAnalysisRecipes
   git pull # to get up-to-date
   git remote add foo git@github.com:davidwhogg/foo.git
   git fetch foo
   git merge foo/master
   git push
   ```

4. Then I delete the foo repo from github so as not to confuse me.
