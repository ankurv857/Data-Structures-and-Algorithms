

##-----------------------------------------------------------
## Set working directory
##-----------------------------------------------------------

getwd()
setwd("~/R/Codes Training/Data")

##-----------------------------------------------------------
## Install a package
##-----------------------------------------------------------

install.packages("sqldf")

##-----------------------------------------------------------
## Load a package
##-----------------------------------------------------------

library(sqldf)

##-----------------------------------------------------------
## Unload a package
##-----------------------------------------------------------

detach(package:sqldf)

##-----------------------------------------------------------
## Access help in R
##-----------------------------------------------------------

?plot           # Open help page on plot function
??plot          # Search for keywords in help manual
help.start()    # Open R help pages


#create data frame
a = c(23,45,12,56,34,78,98,65,43,32,65,85,52,54,65)
b = c(2,3,4,3,5,6,8,2,9,7,6,5,4,2,3)
d = c('a','w','w','a','w','a','w','s','w','a','s','w','s','a','s')

c = data.frame(feature = a, dependent = b)



##-----------------------------------------------------------
## Read a CSV file into R
##-----------------------------------------------------------

## Method 1

train <- read.csv("Train.csv")

## Method 2
install.packages("readr")
library(readr)

train <- read_csv("Train.csv")

## Method 3

install.packages("data.table")
library(data.table)

train <- read.csv("Train.csv")


##Method 4

trans <- read.table(file = "Train.csv", sep = ",",
                    header = TRUE, fill = TRUE, quote = "\"",
                    stringsAsFactors = FALSE,
                    na.strings = c("NA", ""))


View(train)


##-----------------------------------------------------------
## Check the imported dataset
##-----------------------------------------------------------

head(train, n = 10)    # Display top 10 rows

str(train)             # Display the metadata

summary(train)         # Summary statistics for the variables

names(train)           # Displays column names

rownames(train)        # Displays the row names / numbers

##-----------------------------------------------------------
## Access elements of the dataframe
##-----------------------------------------------------------

train[1, 2]     # Get the element from 1st row, 2nd column

train[1, ]      # Get the first row in form of a data frame

train[, 1]      # Get the first column in form of a vector

train[1]        # Get the first column in form of a list

class((train[, 1]))     # Use class function to find the type of data structure 

train[c(4,5), c(1,2)]  # What does this line of code do?

colnames(train)[1] = "ID" # Rename a variable name

##-----------------------------------------------------------
## Subset data frame based on some conditions
##-----------------------------------------------------------

which(train$City == 'Delhi')         

DelhiID <-train[which(train$City == 'Delhi'), ]


# Get all transactions where Sales_Amount of $50 or more

highSales <- train[which(train$Sales >= 50),]

##-----------------------------------------------------------
## Sort a data frame by some column
##-----------------------------------------------------------

order(highSales$Sales)   # Get sorted indices in increasing order

order(-highSales$Sales)  # Get sorted indices in decreasing order

sortedSales <- highSales[order(highSales$Sales),]

##-----------------------------------------------------------
## Identify missing values
##-----------------------------------------------------------

which(is.na(train$Sales))

train$Sales[which(is.na(train$Sales))] <- 0

##-----------------------------------------------------------
## Delete objects from memory
##-----------------------------------------------------------

ls()
rm(sortedSales)

rm(list = ls())



summary(train)

sum(train$Sales)  			# Total Sales

sum(train$Sales, na.rm = T)

###------------------------------------------------------------
### Identify the transactions with maximum value
###------------------------------------------------------------

maxSales <- max(train$Sales, na.rm = T)

maxTrans <- train[which(train$Sales == maxSales), ]

###------------------------------------------------------------
### How many uniques in total 
###------------------------------------------------------------

length(maxTrans$ID)

unique(maxTrans$ID)

###------------------------------------------------------------
### Aggregate functions
###------------------------------------------------------------

countOfCusts <- aggregate(x = train['ID'], 
                          by = train['City'], 
                          FUN = length)

# Average Sales by Item
aggregate(formula = Sales ~ Item, 
          data = train, 
          FUN = mean)

# Average Sales By city & item
aggregate(formula = Sales ~ 
            City + Item, data = train, FUN = mean)

###------------------------------------------------------------
### Create derived variables based on conditions
###------------------------------------------------------------

train$salesFlag <- ifelse(train$Sales > 50, 'High', 'Low')

str(train)

##-----------------------------------------------------------
## Creating a factor variable
##-----------------------------------------------------------

train$salesFlag <- ifelse(train$Sales > 50, 1, 0)

sum(train$salesFlag, na.rm = TRUE)

train$salesFlag <- factor(train$salesFlag)

summary(train$salesFlag)

train$salesFlag <- factor(train$salesFlag, levels= c(1,0), 
                          labels = c("High", "Low"))

summary(train$salesFlag)

###------------------------------------------------------------
### Perform joins in R
###------------------------------------------------------------

price = read.csv("price.csv")

# Full Outer Join
outer <- merge(x = train, y = price, 
               by.x = "Item" , 
               by.y = "Item", 
               all = TRUE)

# Inner Join
inner <- merge(x = train, y = price, 
               by.x = "Item" , 
               by.y = "Item", 
               all = FALSE)

# Left Join
left <- merge(x = train, y = price, 
              by.x = "Item" , 
              by.y = "Item", 
              all.x = TRUE)

# Right Join
right <- merge(x = train, y = price, 
               by.x = "Item" , 
               by.y = "Item", 
               all.y = TRUE)




###------------------------------------------------------------
### Date functions in R 
###------------------------------------------------------------

date()  		# Current date and time

Sys.Date()		# Current date

Sys.time()		# Current date and time with time zone

format(Sys.time(), "%A %B %d, %X %Y")	# Date and time in user defined format


#train$InvDate <- as.Date(train$date, format = "%d-%b-%y") # Convert to date format
#months(trans$InvDate)      # Display months in words
#weekdays(trans$InvDate)    # Display weekdays in words



###------------------------------------------------------------
### SQL in R
###------------------------------------------------------------

install.packages("sqldf")
library(sqldf)

df <- sqldf("select distinct Item from train")

sqldf("select distinct ID as trans 
      from train where Item = 'A2'")

SalesByCustomer <- sqldf("select 
                         Item as Item, 
                         sum(Sales) as Total_Sales
                         from train 
                         group by Item")


##-----------------------------------------------------------
## Flow Control in R
## if...else...if...else
##-----------------------------------------------------------

x <- runif(100)

measure <- "median"

if(measure == "median" ) {
  print(median(x))
} else if (measure == "mean") {
  print(mean(x))
} else {
  print("Wrong Input")
}

##-----------------------------------------------------------
## for Loops
##-----------------------------------------------------------

for ( i in 1:5) {
  print(i)
}

##-----------------------------------------------------------
## while Loops
##-----------------------------------------------------------

i <- 0
while (i < 10) {
  print(i)
  i <- i + 1
}

##-----------------------------------------------------------
## Write a function in R to count the number of odd integers
##-----------------------------------------------------------

oddcount <- function(x) {
  k <- 0  ## Assign the value 0 to k
  for (n in x) {  ## Start a FOR loop for every element in x
    
    if (n %% 2 == 1) k <- k + 1  ## %% is a modulo operator
  }
  return(k)
}

oddcount(c(1,2,3,5,7,9,14))

##-----------------------------------------------------------
## A different way of writing the above function
##-----------------------------------------------------------

oddcount2 <- function(x) {
  k <- 0  ## Assign the value 0 to k
  for (i in 1:length(x)) {  ## The length function gives number of elements in x
    if (x[i] %% 2 == 1) k <- k + 1  ## %% is a modulo operator
  }
  return(k)
}


# Univariate in R

train$bucket = ntile(train$Sales,10)

# Plots in R

hist(train$Sales, col = "brown", 
     main = "Sales",
     xlab = "X",
     ylab = "Frequency")

colors()

# Histogram of the transformed variable

hs <- hist(log(train$Sales), 
           col = "grey", 
           main = "Sales",
           xlab = "X",
           ylab = "Frequency")

# Plotting multiple histograms

#for(i in 2:K){
  hist(train[,i], col = "darkred", 
       main = colnames(train)[i])
}

plot(density(log(train$Sales)), 
     col = "darkred",
     main = "Density Plot")

polygon(density(log(train$Sales)), 
        col = "darkred")

# Plot boxplot to identify outliers
boxplot(train$Sales, 
        col = "darkred",
        main = "Box Plot")

boxplot(train$Sales ~ train$Item, 
        col = "darkred",
        main = "Box Plot on Sales",
        xlab = "Item")


###------------------------------------------------------------
###  Bivariate Analysis
###------------------------------------------------------------

# Check relationships

plot(train$ID, 
     train$Sales, 
     main = "Correlation",
     xlab = "ID", 
     ylab = "Sales", 
     col = "brown", pch = 20)

plot(train$ID, 
     log(train$Sales), 
     main = "Correlation",
     xlab = "Birth Rate",
     ylab = "log(GNI Per Capita)", 
     col = "brown", pch = 20)

pairs(~train$Sales + 
        train$ID  , 
      data = train,
      main = "Scatterplot Matrices", 
      panel = panel.smooth)



# Calculate Pearson's correlation coefficient

cor(train[,c(1,4)])

install.packages("Hmisc")
library(Hmisc)



# Recommendation Engine [Collaborative Filtering] 

Cust_data = 'CID.csv'
Task_data = 'TID.csv'
func = function(Cust_data,Task_data,TaskID)
  
{
  # calling libraries
  
  library(data.table)
  library(plyr)
  library(arules)
  library(reshape)
  
  #import data
  
  CID = fread(Cust_data)
  TID = fread(Task_data)
  
  #create a pivot table to form matrix for recommendation engine
  
  piv_cid = cast(CID, CID ~ SKILLID,value = "RATING",fun.aggregate = mean)
  piv_tid = cast(TID, SKILLID ~ TASKID,value = "IMPORTANCE",fun.aggregate = mean)
  
  # reindex the importance value as  it has inverse proportion
  
  for ( i in 2:25 )
  {
    piv_tid[,i] = (1/piv_tid[,i])
  }
  
  # perform the null value treatment
  
  piv_cid[is.na(piv_cid)] = 0
  piv_tid[is.na(piv_tid)] = 0
  
  # convert to matrix
  
  a = as.matrix(piv_cid)
  b= as.matrix(piv_tid)
  
  # multiply both the matrix
  
  cum = a%*%b
  
  # get the customer with highest index from the final matrix
  
  CustID = which.max(cum[,TaskID])
  return(CustID)
}

func('CID.csv','TID.csv',7)

#dummy variables

install.packages("dummy")
library(dummy)

q= factor(train$City)
a = model.matrix(~q-1)
a1 = data.frame(a)
train = data.frame(train,a1)


