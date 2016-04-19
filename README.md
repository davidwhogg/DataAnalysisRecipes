# Data Analysis Recipes

Chapters from Hogg's non-existent book.

### Authors: ###

(Contributions have come from all of the following.)

* **David W. Hogg**, New York University
* **Jo Bovy**, Institute for Advanced Study
* **Dan Foreman-Mackey**, University of Washington
* **Dustin Lang**, Princeton University

### License: ###

**Copyright 2010, 2011, 2012, 2013 the authors.  All rights reserved.**

If you have interest in using or re-using any of this content, get in
touch with Hogg.

### Style notes: ###

- *tentative:* use "pdf" not "PDF".
- When at the end of the sentence, put the `\note` after the period,
  but when at the end of a phrase, put the `\note` before the comma or
  parenthesis.
- Make sure the endnotes can be read on their own, outside of context.
- Be careful with the words "error", "uncertainty", "probability",
  "frequency", "likelihood".
- Use `()` for function arguments, and `[]` for grouping/precedence.
- Define macros; remember "1, 2, infinity".
- Put new terms in `\emph{}`, put only referred-to words in quotation marks.
- Do in-text itemized lists with `\textsl{(a)}~` and so on.

### Git migration notes: ###

When I want to import stuff from the old SVN repository, I do the
following:

1. I create a new github repository called `foo` and follow the svn
   import instructions.

2. I `git clone` that repository and do things like move the files into
   a directory structure that won't conflict with the current
   structure, like:

        cd
        git clone git@github.com:davidwhogg/foo.git
        cd foo
        mkdir straightline
        git add straightline # I think this is maybe needed?
        git mv *.pdf straightline
        # etc
        # . . .
        git commit -a -m "fixed up directory structure"
        git push

3. I make a subtree merge or something like that (I am new to all this)
   like so:

        cd
        cd DataAnalysisRecipes
        git pull # to get up-to-date
        git remote add foo git@github.com:davidwhogg/foo.git
        git fetch foo
        git merge foo/master
        git push

4. Then I delete the `foo` repo from github so as not to confuse myself.
