PROJECT(thesis)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
FIND_PACKAGE( OpenCV REQUIRED)

#source directory
SET(srcDirectory "src")
#output directory
SET(outDirectory "target")
#essentials directory
SET(essDirectory "essentials")

#main name file
LIST(APPEND fileName "computeChains" "computeChainsAnalysis" "homographie" "mainSeeFlowMap" "motionCompensation_avec_RGMC_Et_stabilisation" "quantityOfSing" "ralentiHisto" "readerVideo" "saliant" "saliantShort" "saturation_Median" "shortenizer" "singularity")

#In this section all the list must finish with "_"

#computeChains linked file name
LIST(APPEND computeChainsList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "rgmc" "Singularity" "_")
#computeChainsAnalysis linked file name
LIST(APPEND computeChainsAnalysisList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "rgmc" "Singularity" "_")
#homographie linked file name
LIST(APPEND homographieList "rgmc" "_")
#mainSeeFlowMap linked file name
LIST(APPEND mainSeeFlowMapList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "Singularity" "_")
#motionCompensation_avec_RGMC_Et_stabilisation linked file name
LIST(APPEND motionCompRGMCList "rgmc" "_")
#quantityOfSing linked file name
LIST(APPEND quantityOfSingList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "_")
#ralentiHisto linked file name
LIST(APPEND ralentiHistoList "_")
#readerVideo linked file name
LIST(APPEND readerVideoList "Field" "NormField" "OpticalFlow" "_")
#saliant linked file name
LIST(APPEND saliantList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "_")
#saliantShort linked file name
LIST(APPEND saliantShortList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "_")
#saturation_Median linked file name
LIST(APPEND saturationMedianList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "rgmc" "_")
#shortenizer linked file name
LIST(APPEND shortenizerList "_")
#singularity linked file name
LIST(APPEND singularityList "Field" "NormField" "OpticalFlow" "Polynome" "projectedOptFlow" "Projection" "_")

#In the same order as "fileName" list
LIST(APPEND extendList ${computeChainsList} ${computeChainsAnalysisList} ${homographieList} ${mainSeeFlowMapList} ${motionCompRGMCList} ${quantityOfSingList} ${ralentiHistoList} ${readerVideoList} ${saliantList} ${saliantShortList} ${saturationMedianList} ${shortenizerList} ${singularityList})

SET(iterator 0)
FOREACH(singleFile ${fileName})
	WHILE(1)
		LIST(GET extendList ${iterator} name)
		MATH(EXPR iterator "${iterator}+1")
		if(${name} MATCHES "_")
			BREAK()
		ENDIF(${name} MATCHES "_")
		LIST(APPEND readerFileList ${essDirectory}/${name}.hpp)
		LIST(APPEND readerFileList ${essDirectory}/${name}.cpp)
	ENDWHILE(1)
	ADD_EXECUTABLE(${outDirectory}/${singleFile} ${readerFileList} ${srcDirectory}/${singleFile}.cpp)
	TARGET_LINK_LIBRARIES(${outDirectory}/${singleFile} ${OpenCV_LIBS})
	SET(readerFileList)
ENDFOREACH(singleFile)

add_definitions(-O3)
add_definitions(-std=c++11)
