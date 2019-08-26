#pragma once

#include <exception>
#include <iostream>


#if defined(_DEBUG)

#define RT_ASSERT(expect_true) if((expect_true) == 0) { __debugbreak(); }
#define RT_ASSERT_PRINT(expect_true, value) if((expect_true) == 0) { std::cout << value << std::endl; __debugbreak(); }

#else

#define RT_ASSERT(expect_true) ;
#define RT_ASSERT_PRINT(expect_true, value) ;

#endif

