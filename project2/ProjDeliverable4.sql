
CREATE DATABASE IMT543_proj_NikhilN
GO 
USE IMT543_proj_NikhilN
GO

---Create tbl_BOOKING---
CREATE TABLE tbl_BOOKING
(BookingID INT NOT NULL Identity(1,1) Primary key,
FlightID INT NOT NULL, 
AirlineID INT NOT NULL, 
TripPurposeID INT NOT NULL, 
Fare MONEY NULL,
TravelDate DATE NOT NULL
);
GO

---Create tbl_TRIP_PURPOSE---
CREATE TABLE tbl_TRIP_PURPOSE
(TripPurposeID INT NOT NULL Identity(1,1) Primary key,
TripPurposeName varchar(50) NOT NULL,
TripPurposeDescr varchar(500) NULL
);
GO 

INSERT INTO tbl_TRIP_PURPOSE
(TripPurposeName)
VALUES ('Vacation'), ('Business'), ('Family'), ('Academic'), ('Multi-Purpose');
GO

---Create tbl_AIRLINE---
CREATE TABLE tbl_AIRLINE
(AirlineID INT NOT NULL Identity(1,1) Primary key,
AirlineName varchar(50) NOT NULL,
AirlineWebsite varchar(500),
AirlinePhoneNumber varchar(50),
AirlineHomeCountryID INT NOT NULL,
AirlineTypeID INT NOT NULL
); 
GO 

---Create tbl_AIRLINE_TYPE---
CREATE TABLE tbl_AIRLINE_TYPE
(AirlineTypeID INT NOT NULL Identity(1,1) Primary key,
AirlineTypeName varchar(50) NOT NULL,
AirlineTypeDescr varchar(500)
);
GO

INSERT INTO tbl_AIRLINE_TYPE
(AirlineTypeName)
VALUES ('Public') , ('Private') , ('State Owned');
GO

---Create tbl_AIRLINE_HOME_COUNTRY---
CREATE TABLE tbl_AIRLINE_HOME_COUNTRY
(AirlineHomeCountryID INT NOT NULL Identity(1,1) Primary key,
AirlineHomeCountryName varchar(50) NOT NULL
); 
GO 

INSERT INTO tbl_AIRLINE_HOME_COUNTRY
(AirlineHomeCountryName)
VALUES ('United States') , ('Germany') , ('France') , ('United Kingdom') , ('United Arab Emirates') , ('Qatar');
GO

---Create tbl_FLIGHT---
CREATE TABLE tbl_FLIGHT
(FlightID INT NOT NULL Identity(1,1) Primary key,
FlightAircraftID INT NOT NULL,
ArrAirportID INT NOT NULL,
DeptAirportID INT NOT NULL, 
FlightClassID INT NOT NULL, 
FlightNumber varchar(50) NOT NULL,
FlightDurationHours NUMERIC(4,2),
FlightMealID INT NOT NULL, 
BookingID INT NOT NULL
);
GO 

---Create tbl_AIRPORT---
CREATE TABLE tbl_AIRPORT
(AirportID INT NOT NULL Identity(1,1) Primary key,
AirportTypeID INT NOT NULL,
AirportName varchar(50) NOT NULL
);
GO 

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_tbl_FLIGHT_ArrAirportID
FOREIGN KEY (ArrAirportID)
REFERENCES tbl_AIRPORT(AirportID)

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_tbl_FLIGHT_DeptAirportID
FOREIGN KEY (DeptAirportID)
REFERENCES tbl_AIRPORT(AirportID)

---Create tbl_AIRPORT_TYPE---
CREATE TABLE tbl_AIRPORT_TYPE
(AirportTypeID INT NOT NULL Identity(1,1) Primary key, 
AirportTypeName varchar(50) NOT NULL,
AirportTypeDescr varchar(500)
);
GO 

INSERT INTO tbl_AIRPORT_TYPE 
(AirportTypeName)
VALUES ('International') , ('Domestic'), ('Military') , ('Commercial') , ('Private');
GO

---Create tbl_FLIGHT_AIRCRAFT---
CREATE TABLE tbl_FLIGHT_AIRCRAFT
(FlightAircraftID INT NOT NULL Identity(1,1) Primary key,
FlightAircraftName varchar(50) NOT NULL,
FlightAircraftDescr varchar(500),
ManufacturerID INT NOT NULL
);
GO 

---Create tbl_MANUFACTURER---
CREATE TABLE tbl_MANUFACTURER
(ManufacturerID INT NOT NULL Identity(1,1) Primary key,
ManufacturerName varchar(50) NOT NULL
);
GO 

INSERT INTO tbl_MANUFACTURER
(ManufacturerName)
VALUES ('Boeing') , ('Airbus') , ('Bombardier') , ('Embraer');
GO

---Create tbl_FLIGHT_CLASS
CREATE TABLE tbl_FLIGHT_CLASS
(FlightClassID INT NOT NULL Identity(1,1) Primary key,
FlightClassName varchar(50) NOT NULL,
FlightClassDescr varchar(500)
);
GO 

INSERT INTO tbl_FLIGHT_CLASS
(FlightClassName)
VALUES ('First Class') , ('Business Class'), ('Economy Class'), ('Economy Plus Class'), ('Basic Class');
GO

---Create tbl_MEAL---
CREATE TABLE tbl_MEAL
(MealID INT NOT NULL Identity(1,1) Primary key,
MealName varchar(50) NOT NULL,
MealDescr varchar(500)
);
GO 

INSERT INTO tbl_MEAL
(MealName)
VALUES ('Breakfast') , ('Brunch'), ('Lunch'), ('Snack'), ('Dinner'), ('Drinks Only');
GO

---Create tbl_FLIGHT_MEAL---
CREATE TABLE tbl_FLIGHT_MEAL
(FlightMealID INT NOT NULL Identity(1,1) Primary key,
FlightID INT NOT NULL,
MealID INT NOT NULL,
FlightMealName varchar(50) NOT NULL,
FlightMealDescr varchar(500)
);
GO

---FOREIGN KEY CONSTRAINTS--
ALTER TABLE tbl_BOOKING
ADD CONSTRAINT FK_Booking_FlightID
FOREIGN KEY (FlightID)
REFERENCES tbl_FLIGHT(FlightID)

ALTER TABLE tbl_BOOKING
ADD CONSTRAINT FK_Booking_AirlineID
FOREIGN KEY (AirlineID)
REFERENCES tbl_Airline(AirlineID)

ALTER TABLE tbl_BOOKING
ADD CONSTRAINT FK_Booking_TripPurposeID
FOREIGN KEY (TripPurposeID)
REFERENCES tbl_TRIP_PURPOSE(TripPurposeID)

ALTER TABLE tbl_AIRLINE
ADD CONSTRAINT FK_Airline_AirlineHomeCountryID
FOREIGN KEY (AirlineHomeCountryID)
REFERENCES tbl_AIRLINE_HOME_COUNTRY(AirlineHomeCountryID)

ALTER TABLE tbl_AIRLINE
ADD CONSTRAINT FK_Airline_AirlineTypeID
FOREIGN KEY (AirlineTypeID)
REFERENCES tbl_AIRLINE_TYPE(AirlineTypeID)

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_Flight_FlightAircraftID
FOREIGN KEY (FLightAircraftID)
REFERENCES tbl_FLIGHT_AIRCRAFT(FlightAircraftID)

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_Flight_FlightClassID
FOREIGN KEY (FlightClassID)
REFERENCES tbl_FLIGHT_CLASS(FlightClassID)

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_Flight_FlightMealID
FOREIGN KEY (FlightMealID)
REFERENCES tbl_FLIGHT_MEAL(FlightMealID)

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT FK_Flight_BookingID
FOREIGN KEY (BookingID)
REFERENCES tbl_BOOKING(BookingID)

ALTER TABLE tbl_AIRPORT
ADD CONSTRAINT FK_Airport_AirportTypeID
FOREIGN KEY (AirportTypeID)
REFERENCES tbl_AIRPORT_TYPE(AirportTypeID)

ALTER TABLE tbl_FLIGHT_AIRCRAFT
ADD CONSTRAINT FK_Flight_Aircraft_ManufacturerID
FOREIGN KEY (ManufacturerID)
REFERENCES tbl_Manufacturer(ManufacturerID)

ALTER TABLE tbl_FLIGHT_MEAL
ADD CONSTRAINT FK_Flight_Meal_FlightID 
FOREIGN KEY (FlightID)
REFERENCES tbl_Flight(FlightID)

ALTER TABLE tbl_FLIGHT_MEAL
ADD CONSTRAINT FK_Flight_Meal_MealID
FOREIGN KEY (MealID)
REFERENCES tbl_MEAL(MealID)
GO

---STORED PROCEDURE FOR FKs in tbl_AIRLINE ---
CREATE PROCEDURE insertTbl_AIRLINE
@AirlineHomeCountryName varchar(50),
@AirlineTypeName varchar(50),
@Name varchar(50),
@Web varchar(500),
@Phone varchar(50)
AS
DECLARE @Country_ID INT, @AT_ID INT 

SET @Country_ID = (SELECT AirlineHomeCountryID
    FROM tbl_AIRLINE_HOME_COUNTRY
    WHERE AirlineHomeCountryID = @Country_ID)

IF @Country_ID IS NULL
   	BEGIN
   	PRINT 'Problem with empty variable; check spelling of all parameters'
   	RAISERROR ('@Country_ID cannot be NULL; process is terminating', 11,1)
   	RETURN
   	END

SET @AT_ID = (SELECT AirlineTypeID
    FROM tbl_AIRLINE_TYPE 
    WHERE AirlineTypeID = @AT_ID)

IF @AT_ID IS NULL
   	BEGIN
   	PRINT 'Problem with empty variable; check spelling of all parameters'
   	RAISERROR ('@AT_ID cannot be NULL; process is terminating', 12,1)
   	RETURN
   	END

BEGIN TRANSACTION
INSERT INTO tbl_AIRLINE (AirlineName, AirlineWebsite, AirlinePhoneNumber, AirlineHomeCountryID, AirlineTypeID)
VALUES (@Name, @Web, @Phone, @Country_ID, @AT_ID)
COMMIT TRANSACTION 
GO 

---STORED PROCEDURE FOR FKs in Tbl_BOOKING
CREATE PROCEDURE insertTbl_BOOKING
@FlightNumber varchar(50),
@AirlineName varchar(50),
@TripPurposeName varchar(50),
@Fare MONEY,
@Date DATE
AS
DECLARE @F_ID INT, @A_ID INT, @T_ID INT

SET @F_ID = (SELECT FlightID
    FROM tbl_FLIGHT
    WHERE FlightID = @F_ID)

IF @F_ID IS NULL
   	BEGIN
   	PRINT 'Problem with empty variable; check spelling of all parameters'
   	RAISERROR ('@F_ID cannot be NULL; process is terminating', 13,1)
   	RETURN
   	END

SET @A_ID = (SELECT AirlineID
    FROM tbl_AIRLINE
    WHERE AirlineID = @A_ID)

IF @A_ID IS NULL
   	BEGIN
   	PRINT 'Problem with empty variable; check spelling of all parameters'
   	RAISERROR ('@A_ID cannot be NULL; process is terminating', 14,1)
   	RETURN
   	END

SET @T_ID = (SELECT TripPurposeID
    FROM tbl_TRIP_PURPOSE
    WHERE TripPurposeID = @T_ID)


IF @T_ID IS NULL
   	BEGIN
   	PRINT 'Problem with empty variable; check spelling of all parameters'
   	RAISERROR ('@T_ID cannot be NULL; process is terminating', 15,1)
   	RETURN
   	END

BEGIN TRANSACTION
INSERT INTO tbl_BOOKING (FlightID, AirlineID, TripPurposeID, Fare, TravelDate)
VALUES (@F_ID, @A_ID, @T_ID, @Fare, @Date)
COMMIT TRANSACTION
GO

---ADD COMPUTED COLUMN to calculate total duration spent traveling to 'DXB' (Dubai) for business purpose---
CREATE FUNCTION fn_TotalBusinessDubaiDuration(@PK INT)
RETURNS INT 
AS
BEGIN 

DECLARE @RET INT = (SELECT SUM(FlightDurationHours)
					FROM tbl_AIRPORT AP 
						JOIN tbl_FLIGHT F ON F.ArrAirportID = AP.AirportID
						JOIN tbl_BOOKING B ON B.FlightID = F.FlightID 
						JOIN tbl_TRIP_PURPOSE TP ON TP.TripPurposeID = B.TripPurposeID
					WHERE F.ArrAirportID = 'DXB'
					AND TP.TripPurposeName = 'Business'
					AND F.FlightID = @PK)
RETURN @RET
END
GO

ALTER TABLE tbl_FLIGHT
ADD Calc_TotalBusinessDubaiDuration AS (fn_TotalBusinessDubaiDuration(FlightID))
GO



---ADD COMPUTED COLUMN to calculate total fare spent on all Business and Academic purpose bookings---
CREATE FUNCTION fn_TotalBusinessAcademicFares(@PK INT)
RETURNS INT
AS 
BEGIN 

DECLARE @RET INT = (SELECT SUM(Fare)
					FROM tbl_BOOKING B
						JOIN tbl_TRIP_PURPOSE TP ON TP.TripPurposeID = B.TripPurposeID
					WHERE TP.TripPurposeName = 'Business'
					OR TP.TripPurposeName = 'Academic'
					AND B.BookingID = @PK)
RETURN @RET
END
GO

ALTER TABLE tbl_BOOKING
ADD Calc_TotalBusinessAcademicFares AS (fn_TotalBusinessAcademicFares(BookingID))
GO


---BUSINESS RULE: No airlines with home countries of North Korea, Cuba, Iran, Syria, Venezuela allowed for booking due to US sanctions.---
CREATE FUNCTION fn_NoSanctionedCountryAirlines()
RETURNS INT 
AS
BEGIN

DECLARE @RET INT = 0
IF EXISTS (SELECT *
			FROM tbl_AIRLINE_HOME_COUNTRY AHC
				JOIN tbl_AIRLINE A ON A.AirlineHomeCountryID = AHC.AirlineHomeCountryID
				JOIN tbl_BOOKING B ON B.AirlineID = A.AirlineID
			WHERE AHC.AirlineHomeCountryName IN ('North Korea', 'Cuba', 'Iran', 'Syria', 'Venezuela'))
			BEGIN	
				SET @RET = 1
			END
RETURN @RET
END
GO

ALTER TABLE tbl_BOOKING
ADD CONSTRAINT NoSanctionedAirlinesinUS
CHECK (dbo.fn_NoSanctionedCountryAirlines() = 0)
GO 

---BUSINESS RULE: Ryanair fleet consists of only Boeing 737-800 aircraft to "keep costs down and safety standards up"---
CREATE FUNCTION fn_RyanAirOnlyBoeing737800()
RETURNS INT 
AS
BEGIN 

DECLARE @RET INT = 0 
IF EXISTS (SELECT *
			FROM tbl_MANUFACTURER M
				JOIN tbl_FLIGHT_AIRCRAFT FA ON FA.ManufacturerID = M.ManufacturerID
				JOIN tbl_FLIGHT F ON F.FlightAircraftID = FA.FlightAircraftID
				JOIN tbl_BOOKING B ON B.FlightID = F.FlightID 
				JOIN tbl_AIRLINE A ON A.AirlineID = B.AirlineID
			WHERE A.AirlineName = 'Ryanair'
			AND M.ManufacturerName <> 'Boeing'
			AND FA.FlightAircraftName <> '737-800')
			BEGIN
				SET @RET = 1 
			END 
RETURN @RET 
END 
GO 

ALTER TABLE tbl_FLIGHT
ADD CONSTRAINT NoNonBoeing737800Ryanair
CHECK (dbo.fn_RyanAirOnlyBoeing737800()=0)
GO

---COMPLEX QUERY: How many distinct lunches were served on Emirates Business class flights departing from DXB (Dubai) and flying an Airbus A380 before March 2020?--- 

SELECT AL.AirlineID, AL.AirlineName, COUNT (DISTINCT FM.FlightMealID) AS Num_Unique_Lunches
	FROM tbl_MEAL M
		JOIN tbl_FLIGHT_MEAL FM ON FM.MealID = M.MealID 
		JOIN tbl_FLIGHT F ON F.FlightMealID = FM.FlightMealID
		JOIN tbl_AIRPORT AP ON AP.AirportID = F.DeptAirportID
		JOIN tbl_FLIGHT_CLASS FC ON FC.FlightClassID = F.FlightClassID
		JOIN tbl_BOOKING B ON B.FlightID = F.FlightID
		JOIN tbl_AIRLINE AL ON AL.AirlineID = B.AirlineID
		JOIN tbl_FLIGHT_AIRCRAFT FA ON FA.FlightAircraftID = F.FlightAircraftID
		JOIN tbl_MANUFACTURER MR ON MR.ManufacturerID = FA.ManufacturerID
	WHERE AL.AirlineName = 'Emirates'
	AND FC.FlightClassName = 'Business'
	AND AP.AirportName = 'DXB'
	AND MR.ManufacturerName = 'Airbus'
	AND FA.FlightAircraftName = 'A380'
	AND B.TravelDate < 'March 1, 2020' 
GROUP BY AL.AirlineID, AL.AirlineName

/*
COMPLEX QUERY: Determine which airlines meet all the following conditions:
1. Flew more than 5 booked flights departing from New York (JFK) between 2015 and 2021
2. Flew at least one flight aboard an Airbus A380.
*/

SELECT AL.AirlineID, AL.AirlineName, COUNT (DISTINCT F.FlightID) AS Num_Unique_Flights
FROM tbl_AIRLINE AL 
JOIN tbl_BOOKING B ON B.AirlineID = AL.AirlineID
JOIN tbl_FLIGHT F ON F.BookingID = B.BookingID 
JOIN tbl_AIRPORT AP ON AP.AirportID = F.DeptAirportID
JOIN (SELECT AL.AirlineID, AL.AirlineName 
	FROM tbl_AIRLINE AL 
	JOIN tbl_BOOKING B ON B.AirlineID = AL.AirlineID
	JOIN tbl_FLIGHT F ON F.BookingID = B.BookingID 
	JOIN tbl_FLIGHT_AIRCRAFT FA ON FA.FlightAircraftID = F.FlightAircraftID
	WHERE FA.FlightAircraftName = 'A380') subq ON subq.AirlineID = AL.AirlineID
WHERE F.DeptAirportID = 'JFK'
AND B.TravelDate BETWEEN '2015' AND '2021'
GROUP BY AL.AirlineID, AL.AirlineName
HAVING COUNT (DISTINCT F.FlightID) > 5



