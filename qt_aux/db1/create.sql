create table patient(
  firstname TEXT,
  middlename TEXT,
  lastname TEXT,
  id INTEGER primary key NOT NULL,
  gender TEXT,
  birthdate TEXT,
  weight REAL,
  height REAL
);

create table images(
  patientid INTEGER,
  filename TEXT,
  FOREIGN KEY(patientid) REFERENCES patient(id)
);

insert into patient values('Manish', '', 'Sharma', 
  1, 'm', '1/1/2020', 143.6, 201.2);

insert into patient values('Ashish', '', 'Avachat', 
  2, 'm', '2/28/1991', 143.6, 201.2);

insert into patient values('Lukas', '', 'Tucker', 
  3, 'm', '3/15/1988', 143.6, 201.2);

insert into patient values('Jonathan', '', 'Scott', 
  4, 'm', '5/2/1991', 143.6, 201.2);

insert into patient values('Edward', 'Thomas', 'Norris', 
  5, 'm', '6/30/1990', 143.6, 201.2);

insert into patient values('Xin', '', 'Liu', 
  6, 'm', '5/4/1977', 143.6, 201.2);

insert into patient values('Gary', '', 'Mueller', 
  7, 'm', '8/1/1870', 143.6, 201.2);

insert into patient values('Carlos', '', 'Castano', 
  8, 'm', '1/1/1900', 143.6, 201.2);

insert into patient values('Ayodegi', '', 'Alajo', 
  9, 'm', '9/16/1950', 143.6, 201.2);

insert into patient values('Hank', '', 'Lee', 
  10, 'm', '11/30/1999', 143.6, 201.2);

insert into patient values('Erica', '', 'Tucker', 
  11, 'f', '12/31/2013', 143.6, 201.2);


insert into images values(5, '/path/to/img1');
insert into images values(5, '/path/to/img2');
insert into images values(5, '/path/to/img3');

insert into images values(6, '/path/to/img1');
insert into images values(6, '/path/to/img2');

.exit







