package com.example.helloworldservice;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(HelloWorldService.HelloWorldController.class)
class HelloWorldServiceTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void testSayHelloReturnsCorrectMessage() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk())
                .andExpect(content().string("Hello, World! This is a test change."));
    }

    @Test
    void testEndpointReturnsHttp200Status() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk());
    }

    @Test
    void testEndpointAccessibleAtCorrectPath() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk())
                .andExpect(content().string("Hello, World! This is a test change."));
    }

    @Test
    void testResponseBodyIsExactlyHelloWorld() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(content().string("Hello, World! This is a test change."));
    }
}
