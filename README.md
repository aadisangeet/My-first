# My-first

A simple Spring Boot Hello World REST service demonstrating the basics of building a RESTful web application.

## Overview

This is a minimal Spring Boot application that exposes a single REST endpoint returning a greeting message. It's designed as an educational example for learning Spring Boot fundamentals.

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher

## Building the Application

To build the project:

```bash
mvn clean package
```

This will create an executable JAR file in the `target/` directory.

## Running the Application

You can run the application in several ways:

### Using Maven:
```bash
mvn spring-boot:run
```

### Using the JAR file:
```bash
java -jar target/hello-world-service-1.0.0.jar
```

The application will start on port 8080 by default.

## Testing the Endpoint

Once the application is running, you can test the endpoint:

```bash
curl http://localhost:8080/api/hello
```

Expected response:
```
Hello, World!
```

## Running Tests

To run the test suite:

```bash
mvn test
```

Or use the provided test script:

```bash
./run-tests.sh
```

## API Documentation

### Endpoints

- **GET /api/hello**
  - Description: Returns a greeting message
  - Response: `Hello, World!` (text/plain)
  - Status Code: 200 OK

## Project Structure

```
src/
├── main/
│   └── java/
│       └── com/example/helloworldservice/
│           └── HelloWorldService.java    # Main application and controller
└── test/
    └── java/
        └── com/example/helloworldservice/
            ├── HelloWorldServiceTest.java        # Unit tests
            └── HelloWorldIntegrationTest.java    # Integration tests
```

## Note

The `llama-benchmark/` directory in this repository contains an unrelated Python benchmarking tool for Llama models. This appears to be in the wrong repository and is not part of the Spring Boot Hello World service.

## Technology Stack

- Spring Boot 3.1.5
- Java 17
- Maven
- JUnit 5
