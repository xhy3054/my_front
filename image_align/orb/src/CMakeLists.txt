# add_library( myorb SHARED myorb.cpp)
# target_link_libraries(myorb ${OpenCV_LIBS})

add_library( ORBextractor SHARED ORBextractor.cpp )
target_link_libraries( ORBextractor ${OpenCV_LIBS} )

add_library( myharris SHARED myharris.cpp )
target_link_libraries( myharris ${OpenCV_LIBS} )
