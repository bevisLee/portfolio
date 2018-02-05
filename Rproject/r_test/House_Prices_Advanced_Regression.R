####_0. Data_description -----------------------
##	No	Variable_Name  Var_Type	Unique_Val Num_NA	Description_eng	etc	Description_kor
##	1	Id          	integer	    2919	0			
##	2	MSSubClass	    integer	    16	    0	Identifies the type of dwelling involved in the Sale.		?????? ???o? ???? ?????????????? ????
##	3	MSZoning	    factor	    6	    4	Identifies the general zoning classification of the sale.	factor ???? ????	?????? ???????? ???? ?????? ????
##	4	LotFrontage	    integer	    129	    486	Linear feet of street connected to property		????Z ?????? ?????? ???? ????
##	5	LotArea     	integer	    1951	0	Lot size in square feet		?????? ????(1???? ????)?? ???? ??????(???? ???? ???????? ?? ???? ???? ????????)
##	6	Street	        factor	    2   	0	Type of road access to property		??????Z ???? ???? ?????????? ????????
##	7	Alley	        factor	    3	    2721	Type of alley access to property	NA  No alley access	????Z ???? ???? ???????? ????????
##	8	LotShape	    factor	    4	    0	General shape of property		?????? ???????? ????
##	9	LandContour	    factor	    4	    0	Flatness of the property		?????? ??????
##	10	Utilities	    factor	    3	    2	Type of utilities available		???? ?????? ???????????? ????????
##	11	LotConfig	    factor	    5	    0	Lot configuration		???? ????
##	12	LandSlope	    factor	    3	    0	Slope of property		?????? ????
##	13	Neighborhood	factor	    25	    0	Physical locations within Ames city limits		Ames ???? ???? ???? ?????? ????????
##	14	Condition1	    factor	    9	    0	Proximity to various conditions		?????? ?????????? ????????
##	15	Condition2	    factor	    8	    0	Proximity to various conditions (if more than one is present)		?????? ?????????? ????????(?? ????)
##	16	BldgType	    factor	    5	    0	Type of dwelling		?????? ????????
##	17	HouseStyle	    factor	    8	    0	Style of dwelling		?????? ??????
##	18	OverallQual	    integer	    10	    0	Rates the overall material and finish of the house		?? ??u?? ???? ?? ???? ????????
##	19	OverallCond	    integer	    9	    0	Rates the overall condition of the house		???? ???????? ???? ????????
##	20	YearBuilt	    integer	    118	    0	Original construction date		???? ???? ????
##	21	YearRemodAdd	integer	    61	    0	Remodel date (same as construction date if no remodeling or additions)		???? ???? (???????? ???? ?????????? ???? ???? ?????? ????)
##	22	RoofStyle	    factor	    6	    0	Type of roof		???? ????????
##	23	RoofMatl	    factor	    8	    0	Roof material		???? ????
##	24	Exterior1st	    factor	    16	    1	Exterior covering on house		???? ????
##	25	Exterior2nd	    factor	    17	    1	Exterior covering on house (if more than one material)		???? ???? (?? ????)
##	26	MasVnrType	    factor	    5	    24	Masonry veneer type		???????? ?????? ????????
##	27	MasVnrArea	    integer	    445	    23	Masonry veneer area in square feet		???????? ?????? ???? (???? ????)
##	28	ExterQual	    factor	    4	    0	Evaluates the quality of the material on the exterior		???? ???? ???? ????
##	29	ExterCond	    factor	    5	    0	Evaluates the present condition of the material on the exterior		???? ???? ???? ???? ????
##	30	Foundation	    factor	    6	    0	Type of foundation		???? ????????
##	31	BsmtQual	    factor	    5	    81	Evaluates the height of the basement	NA No Basement	?????? ???? ????
##	32	BsmtCond	    factor	    5   	82	Evaluates the general condition of the basement	NA No Basement	?????? ???????? ???? ????
##	33	BsmtExposure	factor	    5   	82	Refers to walkout or garden level walls	NA No Basement	?????? ???? : ????????????, ???? ????????
##	34	BsmtFinType1	factor	    7   	79	Rating of basement finished area	NA No Basement	?????? ???? ????(???? ????)
##	35	BsmtFinSF1	    integer	    992 	1	Type 1 finished square feet		????1 : ???? ???? ????
##	36	BsmtFinType2	factor	    7   	80		NA No Basement	
##	37	BsmtFinSF2	    integer	    273	    1	Type 2 finished square feet		????2 : ???? ???? ????
##	38	BsmtUnfSF	    integer	    1136	1	Unfinished square feet of basement area		?????? ?????? ????????(???? ????)
##	39	TotalBsmtSF	    integer	    1059	1	Total square feet of basement area		?????? ?? ????(???? ????)
##	40	Heating	        factor	    6	    0	Type of heating		???? ????????
##	41	HeatingQC	    factor	    5	    0	Heating quality and condition		???? ???? ?? ????
##	42	CentralAir	    factor	    2	    0	Central air conditioning		???? ???? ????????????
##	43	Electrical	    factor	    6	    1	Electrical system		???? ?y???
##	44	X1stFlrSF	    integer	    1083	0			
##	45	X2ndFlrSF	    integer	    635	    0			
##	46	LowQualFinSF	integer	    36	    0	Low quality finished square feet (all floors)		???????? ?????? ?????? ???? ????(???? ??)
##	47	GrLivArea	    integer	    1292	0	Above grade (ground) living area square feet		???? ???? ????
##	48	BsmtFullBath	integer	    5   	2	Basement full bathrooms		?????? : ??u ????
##	49	BsmtHalfBath	integer	    4   	2	Basement half bathrooms		?????? : ???? ????
##	50	FullBath	    integer	    5   	0	Full bathrooms above grade		??u ???? ????
##	51	HalfBath	    integer	    3   	0	Half baths above grade		???? ???? ????
##	52	BedroomAbvGr	integer	    8   	0			
##	53	KitchenAbvGr	integer	    4   	0			
##	54	KitchenQual	    factor	    5   	1	Kitchen quality		???? ????
##	55	TotRmsAbvGrd	integer	    14  	0	Total rooms above grade (does not include bathrooms)		??u ?? ????(???? ????????)
##	56	Functional	    factor	    8   	2	Home functionality (Assume typical unless deductions are warranted)		?? ???? (?????????? ???????? ???? ?? ???????? ????????)
##	57	Fireplaces	    integer	    5   	0	Number of fireplaces		?????? ??
##	58	FireplaceQu	    factor	    6   	1420	Fireplace quality	NA No Fireplace	?????? ????
##	59	GarageType	    factor	    7   	157	Garage location	NA No Garage	???? ????????
##	60	GarageYrBlt	    integer	    104 	159	Year garage was built		???? ???? ????
##	61	GarageFinish	factor	    4   	159	Interior finish of the garage	NA No Garage	???? ???? ??????
##	62	GarageCars	    integer	    7   	1	Size of garage in car capacity		???? ????
##	63	GarageArea	    integer	    604 	1			
##	64	GarageQual	    factor	    6	    159	Garage quality	NA No Garage	???? ????
##	65	GarageCond	    factor	    6   	159	Garage condition	NA No Garage	???? ????????
##	66	PavedDrive	    factor	    3   	0	Paved driveway		???? ????
##	67	WoodDeckSF	    integer	    379 	0	Wood deck area in square feet		???? ???? ????
##	68	OpenPorchSF	    integer	    252 	0	Open porch area in square feet		???? ????
##	69	EnclosedPorch	integer	    183 	0	Enclosed porch area in square feet		???? ?? ???? ????
##	70	X3SsnPorch	    integer	    31  	0			
##	71	ScreenPorch	    integer	    121 	0	Screen porch area in square feet		?????? ???? ????
##	72	PoolArea	    integer	    14	    0	Pool area in square feet		?????? ????
##	73	PoolQC	        factor	    4	    2909	Pool quality	NA No Pool	?????? ????
##	74	Fence	        factor	    5	    2348	Fence quality	NA No Fence	?????? ????
##	75	MiscFeature	    factor	    5   	2814	Miscellaneous feature not covered in other categories	NA None	???? ???????? ?????? ???? ???? ????
##	76	MiscVal	        integer	    38  	0	$Value of miscellaneous feature		???? ?????? ????
##	77	MoSold	        integer	    12	    0	Month Sold (MM)		?? ????
##	78	YrSold	        integer	    5   	0	Year Sold (YYYY)		?? ????
##	79	SaleType	    factor	    10  	1	Type of sale		???? ????????
##	80	SaleCondition	factor	    6   	0	Condition of sale		???? ????????

####_1. Data load and first Prediction export -----------------------
if (!require("data.table")) { install.packages("data.table") }

# load data using fread
train <- fread("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv", stringsAsFactors = T)
test <- fread("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv", stringsAsFactors = T)

#No. of rows and columns in Train, test
dim(train)
dim(test)

#Check train and test
str(train) # 81 variables
str(test) # 80 variables

#combine data set
test[, SalePrice := mean(train$SalePrice)]
c <- list(train, test)
combin <- rbindlist(c) # rbind ???? ????

####_2. EDA : NA data u?? -----------------------
## ???? Kernels - https://www.kaggle.com/bisaria/handling-missing-data
## Impute Missing Data
if (!require("Amelia")) { install.packages("Amelia") }

missmap(combin[, 1:80],
 main = "Missing values in Housing Prices Dataset",
 y.labels = NULL,
 y.at = NULL)

which(sapply(combin[, 1:80], function(x) sum(is.na(x))) > 0)

### Following features have missing values
##	No	Variable_Name  Var_Type	Unique_Val Num_NA	Description_eng	etc	Description_kor
##	73	PoolQC	        factor	    4	    2909	Pool quality	NA No Pool	?????? ????
##	75	MiscFeature	    factor	    5   	2814	Miscellaneous feature not covered in other categories	NA None	???? ???????? ?????? ???? ???? ????
##	7	Alley	        factor	    3	    2721	Type of alley access to property	NA  No alley access	????Z ???? ???? ???????? ????????
##	74	Fence	        factor	    5	    2348	Fence quality	NA No Fence	?????? ????
##	58	FireplaceQu	    factor	    6   	1420	Fireplace quality	NA No Fireplace	?????? ????
##	4	LotFrontage	    integer	    129	    486	    Linear feet of street connected to property		????Z ?????? ?????? ???? ????
##	60	GarageYrBlt	    integer	    104 	159	    Year garage was built		???? ???? ????
##	61	GarageFinish	factor	    4   	159	    Interior finish of the garage	NA No Garage	???? ???? ??????
##	64	GarageQual	    factor	    6	    159	    Garage quality	NA No Garage	???? ????
##	65	GarageCond	    factor	    6   	159	    Garage condition	NA No Garage	???? ????????
##	59	GarageType	    factor	    7   	157	    Garage location	NA No Garage	???? ????????
##	32	BsmtCond	    factor	    5   	82	    Evaluates the general condition of the basement	NA No Basement	?????? ???????? ???? ????
##	33	BsmtExposure	factor	    5   	82	    Refers to walkout or garden level walls	NA No Basement	?????? ???? : ????????????, ???? ????????
##	31	BsmtQual	    factor	    5	    81	    Evaluates the height of the basement	NA No Basement	?????? ???? ????
##	36	BsmtFinType2	factor	    7   	80		Rating of basement finished area (if multiple types)    NA No Basement	
##	34	BsmtFinType1	factor	    7   	79	    Rating of basement finished area	NA No Basement	?????? ???? ????(???? ????)
##	26	MasVnrType	    factor	    5	    24	    Masonry veneer type		???????? ?????? ????????
##	27	MasVnrArea	    integer	    445	    23	    Masonry veneer area in square feet		???????? ?????? ???? (???? ????)
##	3	MSZoning	    factor	    6	    4	    Identifies the general zoning classification of the sale.	factor ???? ????	?????? ???????? ???? ?????? ????
##	10	Utilities	    factor	    3	    2	    Type of utilities available		???? ?????? ???????????? ????????
##	48	BsmtFullBath	integer	    5   	2	    Basement full bathrooms		?????? : ??u ????
##	49	BsmtHalfBath	integer	    4   	2	    Basement half bathrooms		?????? : ???? ????
##	56	Functional	    factor	    8   	2	    Home functionality (Assume typical unless deductions are warranted)		?? ???? (?????????? ???????? ???? ?? ???????? ????????)
##	24	Exterior1st	    factor	    16	    1	    Exterior covering on house		???? ????
##	25	Exterior2nd	    factor	    17	    1	    Exterior covering on house (if more than one material)		???? ???? (?? ????)
##	35	BsmtFinSF1	    integer	    992 	1	    Type 1 finished square feet		????1 : ???? ???? ????
##	37	BsmtFinSF2	    integer	    273	    1	    Type 2 finished square feet		????2 : ???? ???? ????
##	38	BsmtUnfSF	    integer	    1136	1	    Unfinished square feet of basement area		?????? ?????? ????????(???? ????)
##	39	TotalBsmtSF	    integer	    1059	1	    Total square feet of basement area		?????? ?? ????(???? ????)
##	43	Electrical	    factor	    6	    1	    Electrical system		???? ?y???
##	54	KitchenQual	    factor	    5   	1	    Kitchen quality		???? ????
##	62	GarageCars	    integer	    7   	1	    Size of garage in car capacity		???? ????
##	63	GarageArea	    integer	    604 	1	    		
##	79	SaleType	    factor	    10  	1	    Type of sale		???? ????????

####_2-1. PoolQC : Pool quality NA ???? -----------------------
##	73	PoolQC	        factor	    4	    2909	Pool quality	NA No Pool	?????? ????
summary(as.factor(combin$PoolQC))
table(combin$PoolQC)

# PoolArea ?? PoolQC ???? ???????????? ????
table(combin$PoolArea > 0, combin$PoolQC, useNA = "ifany")
# --> ???????????? 13 a?? ?????? 10 ???? ?????? ?? ???? ??????????.

# PoolArea?? NA?? Pool ???????? = 0 ???????? ????
combin[combin$PoolArea == 0,]$PoolQC <- rep('None', 2906)

# ???????? rpart?? ???????? ?????? ?????? ???? ?? ?? ???? ???? ?? ??
# ?????????????? PoolQC?? ???? ???? ???? ?? ?????? ???? ????

# Predict using rpart
if (!require("rpart")) { install.packages("rpart") }

qlty.rpart <- rpart(as.factor(PoolQC) ~ .,
                           data = combin[!is.na(combin$PoolQC), c("YearBuilt", "YearRemodAdd", "PoolQC", "PoolArea", 
                           "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "ExterQual", 
                           "ExterCond", "YrSold", "SaleType", "SaleCondition")], # "X3SsnPorch"
                           method = "class",
                           na.action = na.omit)

combin$PoolQC[is.na(combin$PoolQC)] <- predict(qlty.rpart,
                           combin[is.na(combin$PoolQC), c("YearBuilt", "YearRemodAdd", "PoolQC", "PoolArea",
                           "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "ExterQual",
                           "ExterCond", "YrSold", "SaleType", "SaleCondition")], # "X3SsnPorch"
                           type = "class")

####_2-2. MiscFeature : Miscellaneous feature not covered in other categories NA ???? -----------------------
##	75	MiscFeature	    factor	    5   	2814	Miscellaneous feature not covered in other categories	???? ???????? ?????? ???? ???? ????
# NA None
table(combin$MiscFeature)
table(is.na(combin$MiscFeature)) # NA == TRUE 2,814

# ???? ???? levels?? ???????? ????, NA?????? None?? ??u???? ????
combin[is.na(MiscFeature), "MiscFeature"] <- rep('None', 2814)

####_2-3. Alley : Type of alley access to property NA ???? -----------------------
##	7	Alley	        factor	    3	    2721	Type of alley access to property	????Z ???? ???? ???????? ????????
# NA  No alley access
table(combin$Alley)
table(is.na(combin$Alley)) # NA == TRUE 2,721

# NA?? Alley?? ??????????????, NA?????? None?? ??u???? ????
combin[is.na(Alley), "Alley"] <- rep('None', 2721)

### Fence: Fence quality NA ????
##	74	Fence	        factor	    5	    2348	Fence quality	?????? ????
# NA No Fence
table(combin$Fence)
table(is.na(combin$Fence)) # NA == TRUE 2,348

# NA?? Fence?? ??????????????, NA?????? None?? ??u???? ????
combin[is.na(Fence), "Fence"] <- rep('None', 2348)

####_2-4. FireplaceQu : Fireplace quality NA ???? -----------------------
##	58	FireplaceQu	    factor	    6   	1420	Fireplace quality	?????? ????
# NA No Fireplace
table(combin$FireplaceQu)
table(is.na(combin$FireplaceQu)) # NA == TRUE 1,420

# There are 1420 rows with NAs for  FireplaceQu
# Check number of fireplaces : ?????? ???? ????
table(as.factor(combin$Fireplaces), useNA = "ifany")

# ?????? ???? FireplaceQu ?????????? ????
table(is.na(combin$FireplaceQu), combin$Fireplaces == 0)

# 'No Fireplace' ?????? None?? ??u
combin[is.na(FireplaceQu), "FireplaceQu"] <- rep('None', 1420)

####_2-5. Garage : NA ???? -----------------------
##	60	GarageYrBlt	    integer	    104 	159	    Year garage was built		???? ???? ????
##	61	GarageFinish	factor	    4   	159	    Interior finish of the garage	NA No Garage	???? ???? ??????
##	64	GarageQual	    factor	    6	    159	    Garage quality	NA No Garage	???? ????
##	65	GarageCond	    factor	    6   	159	    Garage condition	NA No Garage	???? ????????
##	59	GarageType	    factor	    7   	157	    Garage location	NA No Garage	???? ????????
##	62	GarageCars	    integer	    7   	1	    Size of garage in car capacity		???? ????
##	63	GarageArea	    integer	    604 	1	    		

table(is.na(combin$GarageType))
table(is.na(combin$GarageArea))
table(is.na(combin$GarageCars))
table(is.na(combin$GarageArea) & is.na(combin$GarageCars)) # ??u ????
table(is.na(combin$GarageYrBlt))
table(is.na(combin$GarageFinish))
table(is.na(combin$GarageQual))
table(is.na(combin$GarageCond))
table(is.na(combin$GarageYrBlt) & is.na(combin$GarageFinish) & is.na(combin$GarageQual) & is.na(combin$GarageCond))

# GarageType = NA, GarageArea ?? GarageCars?? ???? NA?? 1 ??, 
# GarageYrBlt, GarageFinish, GarageQual ?? GarageCond?? NA?? 159 ?? ???? ?? ????, 
# GarageArea & GarageCars = 0 , GarageType NA?? 157 ???? ????
table(combin$GarageArea == 0 & combin$GarageCars == 0 & is.na(combin$GarageType))

# GarageArea = 0???? ???? ?? ???? ?????????? ???? 157 ???? ???? ???????? 
# ???? ???? ???? ???? ???????? '????????' ???????? ????????????????.

combin[combin$GarageArea == 0 & combin$GarageCars == 0 & is.na(combin$GarageType), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond")] <- apply(combin[combin$GarageArea == 0 & combin$GarageCars == 0 & is.na(combin$GarageType), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond")], 2, function(x) x <- rep("None", 157))

# ???? ???? GarageYrBlt, GarageFinish, GarageQual ?? GarageCond ?? ???? ???????? ???? ?????? ????????????

table(is.na(combin$GarageYrBlt) & is.na(combin$GarageFinish) & is.na(combin$GarageQual) & is.na(combin$GarageCond))

combin[is.na(combin$GarageYrBlt) & is.na(combin$GarageFinish) & is.na(combin$GarageQual) & is.na(combin$GarageCond), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond","GarageCars", "GarageArea")]

# Predict GarageArea
area.rpart <- rpart(GarageArea ~ .,
                           data = combin[!is.na(combin$GarageArea),
                           c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")],
                           method = "anova",
                           na.action = na.omit)

combin$GarageArea[is.na(combin$GarageArea)] <- round(predict(area.rpart, combin[is.na(combin$GarageArea), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")]))

# Predict GarageCars
cars.rpart <- rpart(GarageCars ~ .,
                           data = combin[!is.na(combin$GarageCars), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")],
                           method = "anova",
                           na.action = na.omit)

combin$GarageCars[is.na(combin$GarageCars)] <- round(predict(cars.rpart, combin[is.na(combin$GarageCars), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")]))

# Predict GarageYrBlt
blt.rpart <- rpart(as.factor(GarageYrBlt) ~ .,
                           data = combin[!is.na(combin$GarageYrBlt), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")],
                           method = "class",
                           na.action = na.omit)

combin$GarageYrBlt[is.na(combin$GarageYrBlt)] <- as.numeric(as.character(predict(blt.rpart, combin[is.na(combin$GarageYrBlt), c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "YearBuilt", "GarageCars", "GarageArea")], type = "class")))

# ?????? GarageFinish, GarageQual, GarageCond NA ????????
combin[is.na(combin$GarageFinish) & is.na(combin$GarageQual) & is.na(combin$GarageCond),
c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual","GarageCond", "GarageCars", "GarageArea")]

# Predict GarageFinish, GarageQual and GarageCond

# ???? ?????????? ???? ???????????? ?????? ???? 1950 ???? ?????? ???? ???? ????
combin[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950, c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "GarageCars", "GarageArea")]

summary(as.factor(combin$GarageFinish[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950]))

summary(as.factor(combin$GarageQual[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950]))

summary(as.factor(combin$GarageCond[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950]))

# GarageType == "Detchd" & GarageYrBlt == 1950 ???? ???? ????
# GarageFinish == "Unf" / GarageQual == "TA" , GarageCond == "TA"
combin$GarageFinish[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950 & is.na(combin$GarageFinish)] <- "Unf"
combin$GarageQual[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950 & is.na(combin$GarageQual)] <- "TA"
combin$GarageCond[combin$GarageType == "Detchd" & combin$GarageYrBlt == 1950 & is.na(combin$GarageCond)] <- "TA"

####_2-6. MSZoning : Identifies the general NA ???? -----------------------
##	3	MSZoning	    factor	    6	    4	Identifies the general 
table(combin$MSZoning)
table(is.na(combin$MSZoning)) # NA == TRUE 4

combin[is.na(combin$MSZoning),]

msz.rpart <- rpart(as.factor(MSZoning) ~ .,
                           data = combin[!is.na(combin$MSZoning), c("Neighborhood", "Condition1", "Condition2", "MSZoning")],
                           method = "class",
                           na.action = na.omit)

combin$MSZoning[is.na(combin$MSZoning)] <- as.character(predict(msz.rpart, combin[is.na(combin$MSZoning), c("Neighborhood", "Condition1", "Condition2", "MSZoning")], type = "class"))

####_2-7. Basement: NA ???? ----------------------- 
##	32	BsmtCond	    factor	    5   	82	    Evaluates the general condition of the basement	NA No Basement	?????? ???????? ???? ????
##	33	BsmtExposure	factor	    5   	82	    Refers to walkout or garden level walls	NA No Basement	?????? ???? : ????????????, ???? ????????
##	31	BsmtQual	    factor	    5	    81	    Evaluates the height of the basement	NA No Basement	?????? ???? ????
##	36	BsmtFinType2	factor	    7   	80		Rating of basement finished area (if multiple types)    NA No Basement	
##	34	BsmtFinType1	factor	    7   	79	    Rating of basement finished area	NA No Basement	?????? ???? ????(???? ????)
##	48	BsmtFullBath	integer	    5   	2	    Basement full bathrooms		?????? : ??u ????
##	49	BsmtHalfBath	integer	    4   	2	    Basement half bathrooms		?????? : ???? ????
##	38	BsmtUnfSF	    integer	    1136	1	    Unfinished square feet of basement area		?????? ?????? ????????(???? ????)
##	39	TotalBsmtSF	    integer	    1059	1	    Total square feet of basement area		?????? ?? ????(???? ????)

table(is.na(combin$BsmtExposure))
table(is.na(combin$BsmtCond))
table(is.na(combin$BsmtQual))
table(is.na(combin$BsmtFinType2))
table(is.na(combin$BsmtFinType1))
table(is.na(combin$BsmtFinSF1) & is.na(combin$BsmtFinSF2) & is.na(combin$BsmtUnfSF))
table(is.na(combin$BsmtFullBath) & is.na(combin$BsmtHalfBath))
table(combin$TotalBsmtSF == 0 & is.na(combin$BsmtExposure))

combin[is.na(combin$BsmtExposure) & is.na(combin$BsmtCond), c("TotalBsmtSF", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2",
"BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF")]

## TotalBsmtSF ?? ???? ???? 0???? NA?? 2?? ????, 0???????? NA ??u
combin$TotalBsmtSF[is.na(combin$BsmtExposure) & is.na(combin$TotalBsmtSF)] <- 0

# basement data ?? NA?? 79 row ????????, None?? ??u
combin[combin$TotalBsmtSF == 0 & is.na(combin$BsmtExposure), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2")] <-
apply(combin[combin$TotalBsmtSF == 0 & is.na(combin$BsmtExposure), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2")], 2, function(x) x <- rep("None", 79))

# basement Missing data u??
combin[is.na(combin$BsmtExposure) | is.na(combin$BsmtCond) | is.na(combin$BsmtQual) | is.na(combin$BsmtFinType2),
   c("BsmtExposure", "BsmtCond", "BsmtQual","BsmtFinType1", "BsmtFinType2","TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF")]

# ???? basement ?????????? ???? ?????? ?????? basement Missing data ???????? ???????? ?????? ?? ?? ?????? ????????
BsmtFinType2.rpart <- rpart(as.factor(BsmtFinType2) ~ .,
                           data = combin[!is.na(combin$BsmtFinType2), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                           method = "class",
                           na.action = na.omit)

combin$BsmtFinType2[is.na(combin$BsmtFinType2)] <- as.character(predict(BsmtFinType2.rpart,
                                      combin[is.na(combin$BsmtFinType2), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                                                   type = "class"))

BsmtQual.rpart <- rpart(as.factor(BsmtQual) ~ .,
                           data = combin[!is.na(combin$BsmtQual), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                           method = "class",
                           na.action = na.omit)

combin$BsmtQual[is.na(combin$BsmtQual)] <- as.character(predict(BsmtQual.rpart,
                                           combin[is.na(combin$BsmtQual), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                                           type = "class"))

BsmtCond.rpart <- rpart(as.factor(BsmtCond) ~ .,
                           data = combin[!is.na(combin$BsmtCond), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                           method = "class",
                           na.action = na.omit)

combin$BsmtCond[is.na(combin$BsmtCond)] <- as.character(predict(BsmtCond.rpart,
                                           combin[is.na(combin$BsmtCond), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                                                   type = "class"))

BsmtExposure.rpart <- rpart(as.factor(BsmtExposure) ~ .,
                           data = combin[!is.na(combin$BsmtExposure), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                           method = "class",
                           na.action = na.omit)

combin$BsmtExposure[is.na(combin$BsmtExposure)] <- as.character(predict(BsmtExposure.rpart,
                                      combin[is.na(combin$BsmtExposure), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt")],
                                                   type = "class"))

# NA ?? u??
combin[is.na(combin$BsmtFinSF1) | is.na(combin$BsmtFinSF2) | is.na(combin$BsmtUnfSF), c("BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF", "YearBuilt", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath")]

# TotalBsmtSF = 0 ?? ???????? / BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,BsmtFullBath,BsmtHalfBath 0 ???????? ??u
combin$BsmtFinSF1[is.na(combin$BsmtFinSF1) | is.na(combin$BsmtFinSF2) | is.na(combin$BsmtUnfSF)] <- 0
combin$BsmtFinSF2[is.na(combin$BsmtFinSF1) | is.na(combin$BsmtFinSF2) | is.na(combin$BsmtUnfSF)] <- 0
combin$BsmtUnfSF[is.na(combin$BsmtFinSF1) | is.na(combin$BsmtFinSF2) | is.na(combin$BsmtUnfSF)] <- 0
combin$BsmtFullBath[combin$TotalBsmtSF == 0 & is.na(combin$BsmtFullBath)] <- rep(0, 2)
combin$BsmtHalfBath[combin$TotalBsmtSF == 0 & is.na(combin$BsmtHalfBath)] <- rep(0, 2)

####_2-8. Masonry veneer : NA ???? ----------------------- 
##	26	MasVnrType	    factor	    5	    24	    Masonry veneer type		???????? ?????? ????????
##	27	MasVnrArea	    integer	    445	    23	    Masonry veneer area in square feet		???????? ?????? ???? (???? ????)
summary(as.factor(combin$MasVnrType))

# MasVnrType?? NA : 24?? / None : 1,742?? ????????
# MasVnrArea?? None?? 0 ???? NA???? u??

table(combin$MasVnrArea[combin$MasVnrType == "None"])

# 0?? 1,735?? ????????????, ?????? ???? ?????? ???????? ??o?? ????
# MasVnrAreaseem?? 1???????? ?????? ???? ???????? ??????,
# MasVnrArea = 1 --> 0 ???????? ????

combin$MasVnrArea <- ifelse(combin$MasVnrArea == 1, 0, combin$MasVnrArea)

# MasVnrArea > 0 ???? MasVnrType ???? None ???????? NA?? ??u

combin$MasVnrType[combin$MasVnrArea > 0 & combin$MasVnrType == "None" & !is.na(combin$MasVnrType)] <- rep(NA, 4)

table(is.na(combin$MasVnrType) & is.na(combin$MasVnrArea))

# MasVnrType & MasVnrArea ?? NA 23?? ????????
# MasVnrType ?? None?? ??u
# MasVnrArea ?? 0???????? ??u
combin$MasVnrArea[is.na(combin$MasVnrArea)] <- rep(0, 23)
combin$MasVnrType[is.na(combin$MasVnrType) & combin$MasVnrArea == 0] <- rep("None", 23)

# ??u ???? u??
table(combin$MasVnrType, combin$MasVnrArea == 0, useNA = "ifany")

table(combin$MasVnrType, combin$MasVnrArea > 0, useNA = "ifany")

# MasVnrType ?? BrkFace : 2?? , Stone : 1?? 0 ?? ????
# 0 ???? None?? ??u
combin$MasVnrType[combin$MasVnrType == "BrkFace" & combin$MasVnrArea == 0] <- rep("None", 2)

combin$MasVnrType[combin$MasVnrType == "Stone" & combin$MasVnrArea == 0] <- rep("None", 1)

# Predict MasVnrType 
Type.rpart <- rpart(as.factor(MasVnrType) ~ MasVnrArea,
                    data = combin[!is.na(combin$MasVnrType), c("MasVnrType", "MasVnrArea")],method = "class", na.action = na.omit)

combin$MasVnrType[is.na(combin$MasVnrType)] <- as.character(predict(Type.rpart, combin[is.na(combin$MasVnrType), c("MasVnrType", "MasVnrArea")],type = "class"))

####_2-9. Functional : functionality (Assume typical unless deductions are warranted) NA ???? ----------------------- 
##	56	Functional	    factor	    8   	2	    Home functionality (Assume typical unless deductions are warranted)		?? ???? (?????????? ???????? ???? ?? ???????? ????????)
table(combin$Functional)
table(is.na(combin$Functional)) # NA == TRUE 2

# Functional missing data 2 NA ???????? 
combin[is.na(combin$Functional),]

# Likely predictors
func.rpart <- rpart(as.factor(Functional) ~ .,
                           data = combin[!is.na(combin$Functional), c("OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "SaleType", "SaleCondition", "Functional")],
                           method = "class",
                           na.action = na.omit)

combin$Functional[is.na(combin$Functional)] <- as.character(predict(func.rpart, combin[is.na(combin$Functional), c("OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "SaleType", "SaleCondition", "Functional")], type = "class"))

####_2-10. Utilities : Type of utilities available NA ???? ----------------------- 
##	10	Utilities	    factor	    3	    2	    Type of utilities available		???? ?????? ???????????? ????????
table(combin$Utilities)
table(is.na(combin$Utilities)) # NA == TRUE 2

# Utilities missing data 2 NA ???????? 
combin[is.na(combin$Utilities),]

# Likely predictors
util.rpart <- rpart(as.factor(Utilities) ~ .,
                           data = combin[!is.na(combin$Utilities), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "SaleType", "SaleCondition", "Functional", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolArea", "Utilities")], method = "class", na.action = na.omit) # "X3SsnPorch"

combin$Utilities[is.na(combin$Utilities)] <- as.character(predict(util.rpart, combin[is.na(combin$Utilities), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "SaleType", "SaleCondition", "Functional", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolArea", "Utilities")], type = "class"))

####_2-11. Exterior : NA ???? ----------------------- 
##	24	Exterior1st	    factor	    16	    1	    Exterior covering on house		???? ????
##	25	Exterior2nd	    factor	    17	    1	    Exterior covering on house (if more than one material)		???? ???? (?? ????)
table(combin$Exterior1st)
table(combin$Exterior2nd)

table(is.na(combin$Exterior1st) & is.na(combin$Exterior2nd)) # NA == TRUE 1

# Likely predictors

# Predict Exterior1st
ext1.rpart <- rpart(as.factor(Exterior1st) ~ .,
                           data = combin[!is.na(combin$Exterior1st), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond")],
                           method = "class",
                           na.action = na.omit)

combin$Exterior1st[is.na(combin$Exterior1st)] <- as.character(predict(ext1.rpart, combin[is.na(combin$Exterior1st), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond")], type = "class")) # Wd Sdng : 411 -> 412

# Predict Exterior2nd
ext2.rpart <- rpart(as.factor(Exterior2nd) ~ .,
                           data = combin[!is.na(combin$Exterior2nd), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond")],
                           method = "class",
                           na.action = na.omit)

combin$Exterior2nd[is.na(combin$Exterior2nd)] <- as.character(predict(ext2.rpart, combin[is.na(combin$Exterior2nd), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond")], type = "class")) # Wd Sdng : 391 -> 392

####_2-12. Electrical : Electrical system NA ???? ----------------------- 
##	43	Electrical	    factor	    6	    1	    Electrical system		???? ?y???
table(combin$Electrical)
table(is.na(combin$Electrical)) # NA == TRUE 1

# Likely predictors

# Predict Electrical
elec.rpart <- rpart(as.factor(Electrical) ~ .,
                           data = combin[!is.na(combin$Electrical), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "Electrical")],
                           method = "class",
                           na.action = na.omit)

combin$Electrical[is.na(combin$Electrical)] <- as.character(predict(elec.rpart, combin[is.na(combin$Electrical), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "Electrical")], type = "class")) # SBrkr : 2,671 -> 2,672

####_2-13. KitchenQual : Kitchen quality NA ???? ----------------------- 
##	54	KitchenQual	    factor	    5   	1	    Kitchen quality		???? ????
table(combin$KitchenQual)
table(is.na(combin$KitchenQual)) # NA == TRUE 1

# Likely predictors

# Predict KitchenQual
kit.rpart <- rpart(as.factor(KitchenQual) ~ .,
                           data = combin[!is.na(combin$KitchenQual), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "KitchenQual")],
                           method = "class",
                           na.action = na.omit)

combin$KitchenQual[is.na(combin$KitchenQual)] <- as.character(predict(kit.rpart, combin[is.na(combin$KitchenQual), c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "KitchenQual")], type = "class")) # TA : 1,492 -> 1,493

####_2-14. LotFrontage : Linear feet of street connected to property NA ???? ----------------------- 
##	4	LotFrontage	    integer	    129	    486	    Linear feet of street connected to property		????Z ?????? ?????? ???? ????
table(combin$LotFrontage)
table(is.na(combin$LotFrontage)) # NA == TRUE 486

# Likely predictors

# Predict LotFrontage
frntage.rpart <- rpart(LotFrontage ~ .,
                           data = combin[!is.na(combin$LotFrontage), c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "HouseStyle", "YrSold", "SaleType", "SaleCondition")],
                           method = "anova",
                           na.action = na.omit)

combin$LotFrontage[is.na(combin$LotFrontage)] <- ceiling(predict(frntage.rpart, combin[is.na(combin$LotFrontage), c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "HouseStyle", "YrSold", "SaleType", "SaleCondition")])) # 24 : 49 -> 57, 41 : 14 -> 49, 59 : 27 -> 59, 61 : 17 -> 69, 67 : 22 -> 120, 74 : 39 -> 129, 85 : 76 -> 179, 93 : 13 -> 48, 113 : 3 -> 26, 136 : 1 -> 11

####_2-15. SaleType : Type of sale NA ???? ----------------------- 
##	79	SaleType	    factor	    10  	1	    Type of sale		???? ????????
table(combin$SaleType)
table(is.na(combin$SaleType)) # NA == TRUE 1

# Likely predictors
col.pred <- c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch", "PoolQC", "YrSold", "SaleType", "SaleCondition")

# Predict SaleType
sale.rpart <- rpart(as.factor(SaleType) ~ .,
                           data = combin[!is.na(combin$SaleType), c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolQC", "YrSold", "SaleType", "SaleCondition")],
                           method = "class",
                           na.action = na.omit) # "X3SsnPorch"

combin$SaleType[is.na(combin$SaleType)] <- as.character(predict(sale.rpart, combin[is.na(combin$SaleType), c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolQC", "YrSold", "SaleType", "SaleCondition")], type = "class")) # "X3SsnPorch"
# WD : 2,525 -> 2,526

sapply(combin[, 1:80], function(x) sum(is.na(x)))

# rpart ???? ????????
rm(area.rpart,blt.rpart, BsmtCond.rpart, BsmtExposure.rpart,
    BsmtFinType2.rpart, BsmtQual.rpart, c, cars.rpart, elec.rpart, ext1.rpart, ext2.rpart, frntage.rpart, func.rpart, kit.rpart, msz.rpart, qlty.rpart, sale.rpart, Type.rpart, util.rpart)

####_3. Data Regression Step ------------------------
str(combin)

step(lm(SalePrice ~ ., data = combin), direction="both")

model_fit <- lm(formula = SalePrice ~ MSSubClass + LotArea + LotShape + LandContour +
    LandSlope + Neighborhood + Condition2 + OverallQual + OverallCond +
    YearBuilt + RoofMatl + MasVnrType + MasVnrArea + ExterQual +
    BsmtExposure + BsmtFinSF1 + BsmtFinType2 + BsmtFinSF2 + BsmtUnfSF +
    Heating + CentralAir + `1stFlrSF` + `2ndFlrSF` + FullBath +
    BedroomAbvGr + KitchenAbvGr + KitchenQual + TotRmsAbvGrd +
    GarageYrBlt + GarageArea + PavedDrive + ScreenPorch + PoolArea +
    PoolQC + MiscFeature + MiscVal + SaleCondition, data = combin)

summary(model_fit)

####_4-1. H2O use Modeling ------------------------
# Regression: Starters Guide to Regression - https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
# Random Forest, GBM:Starters Guide to Tree Based Algorithms - https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# Deep Learning:Starters Guide to Deep Learning - https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

#Divide into train and test
c.train <- combin[1:nrow(train),] # 1,460
c.test <- combin[-(1:nrow(train)),] # 1,459

if (!require("h2o")) { install.packages("h2o") }

# h2o ???????? ????
localH2O <- h2o.init(nthreads = -1)

# h2o ???????? ???? ????
h2o.init()

#data to h2o cluster : ?????????? ?????? ????
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)

#check column index number
colnames(train.h2o)

#dependent variable (SalePrice)
y.dep <- 81

#independent variables (dropping ID variables)
x.indep <- c(2:80)



####_4-2. Multiple Regression in H2O ------------------------
regression.model <- h2o.glm(y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")

h2o.performance(regression.model)
## x.indep c(2,5,8,9,12,13,15,18,19,20,23,26,27,28,33,35,36,37,38,40,42,44,45,50,52,53,54,55,60,63,66,71,72,73,75,76,80)
# H2ORegressionMetrics:glm
# ** Reported on training data. **

# MSE:6302411209
# R2:0.0006940737
# Mean Residual Deviance:6302411209
# Null Deviance:9.207911e+12
# Null D.o.F.:1459
# Residual Deviance:9.20152e+12
# Residual D.o.F.:1451
# AIC:37107.03

## x.indep c(2:80)
# H2ORegressionMetrics:glm
# ** Reported on training data. **

# MSE:16710559
# R2:0.3261545
# Mean Residual Deviance:16710559
# Null Deviance:1.353804e+13
# Null D.o.F.:545914
# Residual Deviance:9.122545e+12
# Residual D.o.F.:545898
# AIC:10628689

#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(Id = test$Id, SalePrice = predict.reg$predict)

write.csv(sub_reg, file = "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_reg.csv", row.names = F)

####_4-3. Random Forest in H2O ------------------------
#Random Forest
system.time(
rforest.model <- h2o.randomForest(y = y.dep, x = x.indep, training_frame = train.h2o, ntrees = 800, mtries = 3, max_depth = 4, seed = 1122)
)
# ?????? ?y??? elapsed
# 0.26   0.00   6.41

h2o.performance(rforest.model)
## x.indep c(2,5,8,9,12,13,15,18,19,20,23,26,27,28,33,35,36,37,38,40,42,44,45,50,52,53,54,55,60,63,66,71,72,73,75,76,80)
# H2ORegressionMetrics:drf
# ** Reported on training data. **
# Description:Metrics reported on Out - Of - Bag training samples

# MSE:1574301008
# R2:0.7503799
# Mean Residual Deviance:1574301008

## x.indep c(2:80)
# H2ORegressionMetrics:drf
# ** Reported on training data. **
# Description:Metrics reported on Out - Of - Bag training samples

# MSE:1675893585
# R2:0.7342715
# Mean Residual Deviance:1675893585

#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))
# ?????? ?y??? elapsed
# 0.03   0.01   0.64

#writing submission file
sub_rf <- data.frame(Id = test$Id, SalePrice = predict.reg$predict)
write.csv(sub_rf, file = "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_rf.csv", row.names = F)

####_4-4. GBM in H2O ------------------------
#GBM
system.time(
gbm.model <- h2o.gbm(y = y.dep, ## target: using the logged variable created earlier
                     x = x.indep, ## this can be names or column numbers
                     training_frame = train.h2o, ## H2O frame holding the training data
                     ntrees = 1000, ## use fewer trees than default (50) to speed up training
                     max_depth = 4,
                     learn_rate = 0.01, ## lower learn_rate is better, but use high rate to offset few trees
                     seed = 1122)
)
# ?????? ?y??? elapsed
# 0.27   0.02   9.38

h2o.performance(gbm.model)
## x.indep c(2,5,8,9,12,13,15,18,19,20,23,26,27,28,33,35,36,37,38,40,42,44,45,50,52,53,54,55,60,63,66,71,72,73,75,76,80)
# H2ORegressionMetrics:gbm
# ** Reported on training data. **

# MSE:189978719
# R2:0.9698771
# Mean Residual Deviance:189978719

## x.indep c(2:80)
# H2ORegressionMetrics:gbm
# ** Reported on training data. **

# MSE:146674103
# R2:0.9767435
# Mean Residual Deviance:146674103

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(Id = test$Id, SalePrice = predict.reg$predict)
write.csv(sub_gbm, file = "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_gbm.csv", row.names = F)

####_4-5. Deep-Learning in H2O ------------------------
#deep learning models
system.time(
             dlearning.model <- h2o.deeplearning(y = y.dep,
             x = x.indep,
             training_frame = train.h2o,
             epoch = 60,
             hidden = c(100, 100),
             activation = "Rectifier",
             seed = 1122
             )
)
# ?????? ?y??? elapsed
# 0.37   0.00   5.46

h2o.performance(dlearning.model)
## x.indep c(2,5,8,9,12,13,15,18,19,20,23,26,27,28,33,35,36,37,38,40,42,44,45,50,52,53,54,55,60,63,66,71,72,73,75,76,80)
# H2ORegressionMetrics:deeplearning
# ** Reported on training data. **
# Description:Metrics reported on temporary(load - balanced) training frame

# MSE:170497052
# R2:0.9729661
# Mean Residual Deviance:170497052

## x.indep c(2:80)
# H2ORegressionMetrics:deeplearning
# ** Reported on training data. **
# Description:Metrics reported on temporary(load - balanced) training frame

# MSE:51678823
# R2:0.9918058
# Mean Residual Deviance:51678823

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(Id = test$Id, SalePrice = predict.reg$predict)
write.csv(sub_dlearning, file = "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_dlearning_new.csv", row.names = F)

####_5. Next To Do ------------------------
# GBM, Deep Learning ?? Random Forest???? ???? ???? ?????????? ?????????y?.
# ???? ???? ?????? ?????? ?????????? ?????????y?. H2O?????? ?????????? ???????? h2o.grid???? ???? ?????? ????????????.
# ?????? ?????? ?????????? ?????????? ?????? ?? ???????? ?????????? ?????? ?????????? ???????????y?.
# ??????????????, ?? ???????? ?????????? ???? ???????? ???? ?????? ?????? o???????y?.
