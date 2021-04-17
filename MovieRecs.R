# installing required libraries\
library(tidyverse)
library(dslabs)
library(tidytext)
library(dplyr)
library(caret)
library(matrixStats)
library(data.table)
library(stringr)
library(lubridate)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)


test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# splits edx dataset into training and test sets
options(digits=5)
set.seed(1)
val <- validation
t_ind <- createDataPartition(y=edx$rating, times=1, p=0.05, list=FALSE)
train_x <- edx[-t_ind, ]
test_x <- edx[t_ind, ]

# convert timestamp column to year
train_x$timestamp <- as_datetime(train_x$timestamp) %>% format('%Y')
train_x$timestamp <- as.numeric(train_x$timestamp)
test_x$timestamp <- as_datetime(test_x$timestamp) %>% format('%Y')
test_x$timestamp <- as.numeric(test_x$timestamp)
val$timestamp <- as_datetime(val$timestamp) %>% format('%Y')
val$timestamp <- as.numeric(val$timestamp)


# creating RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu_hat <- mean(edx$rating)

# using regularization to derive optimum value of lambda on edx dataset
lambdas <- seq(3.5, 6, 0.10)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_x$rating)
  
  b_i <- train_x %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_x %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_t <- train_x %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>% group_by(timestamp) %>%
    summarize(b_t = sum(rating-mu-b_i-b_u)/(n()+l))
  
  b_g <- train_x %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by='timestamp') %>%
    group_by(genres) %>%
    summarize(b_g=sum(rating-mu_hat-b_i-b_u-b_t)/(n()+l))
  
  predicted_ratings <-
    test_x %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by='timestamp') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    pull(pred)
  
  predicted_ratings[is.na(predicted_ratings)]=mu_hat-0.5
  
  return(RMSE(test_x$rating,predicted_ratings))
})
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]


l <- 4.8

#modelling movie effects
movie_avgs <- train_x %>% group_by(movieId) %>% summarize(b_i=sum(rating-mu_hat)/(n()+l))
train_x <- train_x %>% left_join(movie_avgs, by='movieId')
#train_x[is.na(train_x)]=0

#modelling user effects
user_avgs <- train_x %>% group_by(userId) %>% summarize(count=n(), b_u=sum(rating-mu_hat-b_i)/(n()+l))
user_avgs <- user_avgs[user_avgs$count>10,]
train_x <- train_x %>% left_join(user_avgs, by='userId')
train_x$b_u[is.na(train_x$b_u)] <- 0

#modelling year effects
year_avgs <- train_x %>% group_by(timestamp) %>% summarize(b_t=sum(rating-mu_hat-b_i-b_u)/(n()+l))
train_x <- train_x %>% left_join(year_avgs, by='timestamp')
train_x$b_t[is.na(train_x$b_t)] <- 0

#modelling genre effects
genre_avgs <- train_x %>% group_by(genres) %>% summarize(count=n(), b_g=sum(rating-mu_hat-b_i-b_u-b_t)/(n()+l))
genre_avgs <- genre_avgs[genre_avgs$count > 500,]
train_x <- train_x %>% left_join(genre_avgs, by='genres')
train_x$b_g[is.na(train_x$b_g)] <- 0


# predict ratings on test set
predicted_ratings <- test_x %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(year_avgs, by='timestamp') %>%
  left_join(genre_avgs, by='genres') 
  

predicted_ratings$b_i[is.na(predicted_ratings$b_i)] <- 0
predicted_ratings$b_u[is.na(predicted_ratings$b_u)] <- 0
predicted_ratings$b_t[is.na(predicted_ratings$b_t)] <- 0
predicted_ratings$b_g[is.na(predicted_ratings$b_g)] <- 0

predicted_ratings <- predicted_ratings %>% mutate(pred= mu_hat + b_i+ b_u + b_t + b_g) 

predicted_ratings$pred[is.na(predicted_ratings$pred)] <- mu_hat
RMSE(test_x$rating, predicted_ratings$pred)

# predict ratings on validation set
predicted_ratings <- val %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='timestamp') %>%
  left_join(genre_avgs, by='genres') 


predicted_ratings$b_i[is.na(predicted_ratings$b_i)] <- 0
predicted_ratings$b_u[is.na(predicted_ratings$b_u)] <- 0
predicted_ratings$b_t[is.na(predicted_ratings$b_t)] <- 0
predicted_ratings$b_g[is.na(predicted_ratings$b_g)] <- 0

predicted_ratings <- predicted_ratings %>% mutate(pred= mu_hat + b_i+ b_u + b_t + b_g) 

predicted_ratings$pred[is.na(predicted_ratings$pred)] <- mu_hat - 0.2
RMSE(val$rating, predicted_ratings$pred)






















