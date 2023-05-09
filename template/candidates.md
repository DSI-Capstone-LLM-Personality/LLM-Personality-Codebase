# Candidate Templates

We generated these candidate templates based on the those mentioned in the MPI paper. However, we are interested in investigating the following aspects:

1. Will **lower-cased** templates be better? **[2 possibilities]**

- `lc`: stands for "**l**ower-**c**ased" options.
- `og`: original version with upper case letter.

2. Will **structured** templates help the model to understand the prompt? Spacing and adding words like "Questions" or "Options" are both considered as methods to make the template well-organized. **[2 possibilities]**

- `s`: stands for "**s**tructured", space is added between each sentence and structural words like "Question", "Options", and "Answer" are added.
- `ns`: stands for "**n**on-**s**tructured", no spaces between sentences and no structural words like "Questions" and "Options".

Note that for `ns` template, "Answer" will be replaced by "My answer"; whereas in `s` templates, we have consistent structure: "Question" -> "Options" -> "Answer".

3. Will different **prompt styles** result in different results? **[3 possibilities]**

- `type-i`:

  > _Given a description of you: "You {}." What do you think?_

- `type-ii`:
  > _Given a description of you: "You {}." What do you think?\
  > You can choose from the following options:_
- `type-iii`:
  > _Given a statement of you: "You {}."\
  > Please choose from the following options to identify how accurately this statement describes you._

4. Will adding more guidances result in different results? **[3 possibilities]**

- `ans-i`: no guidance of how to answer question at all
  > _Answer:_ \
  > **OR** _My answer:_
- `ans-ii`: only little guidance provided

  > _Answer: I choose option_ \
  > **OR** _My answer: I choose option_

- `ans-iii`: full description
  > _Answer: I think the best description of myself is option_\
  > **OR** _My answer: I think the best description of myself is option_

With this be said, we will have $36$ different templates to examine for each model.

## Examples

---

Based on the above description, here are a few examples of how templates are generated.

### Example 1.

Template name: _[og]-[ns]-[type-ii]-[ans-iii].txt_

> _Given a description of you: "You {}." What do you think?\
> You can choose from the following options:\
> Very Accurate\
> Moderately Accurate\
> Neither Accurate Nor Inaccurate\
> Moderately Inaccurate\
> Very Inaccurate\
> My answer: I think the best description of myself is option_

### Example 2.

Template name: _[og]-[s]-[type-ii]-[ans-iii].txt_

> _Question_:
>
> _Given a description of you: "You {}." What do you think?_
>
> _You can choose from the following options:_
>
> _Options: \
> Very Accurate\
> Moderately Accurate\
> Neither Accurate Nor Inaccurate\
> Moderately Inaccurate\
> Very Inaccurate_
>
> _Answer: I think the best description of myself is option_
