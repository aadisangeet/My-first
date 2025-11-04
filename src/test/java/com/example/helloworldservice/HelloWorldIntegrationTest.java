package com.example.helloworldservice;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class HelloWorldIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void testApplicationContextLoads() {
    }

    @Test
    void testFullEndToEndHttpRequestResponse() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk())
                .andExpect(content().string("Hello, World!"));
    }

    @Test
    void testContentTypeHeaderIsCorrect() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk())
                .andExpect(header().exists("Content-Type"))
                .andExpect(content().string("Hello, World!"));
    }

    @Test
    void testSpringBootApplicationStartsCorrectly() throws Exception {
        mockMvc.perform(get("/api/hello"))
                .andExpect(status().isOk());
    }
}
