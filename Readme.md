# Machine Learning Scripts
The purpose of this repo is to learn about machine learning and store all code related
to how we train models and create models.


## projects

### book-reviews
- Data from amazon is converted into a usable form, to feed into linear regression models

- How to use?

You can run the python scripts directly, or you can use docker to run them.

- will convert one file to the usable csv to feed into our model

```
python csv_converter ${filename}
```

- will convert all csvs in a /data folder or you can pass in the filename to convert as well. This should
be ran with docker, so you don't need to write to /data on your local machine.

```
docker run -it -v book-review-data:/data  quay.io/turner/book-review-converter -p polar
```
