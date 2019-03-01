#pragma once

#include <exception>
#include <iostream>

// #define RT_ASSERT(expect_true, value) ;
// #define RT_ASSERT(expect_true) ;

#define RT_ASSERT(expect_true) if((expect_true) == 0) { __debugbreak(); }
#define RT_ASSERT_PRINT(expect_true, value) if((expect_true) == 0) { std::cout << value << std::endl; __debugbreak(); }
