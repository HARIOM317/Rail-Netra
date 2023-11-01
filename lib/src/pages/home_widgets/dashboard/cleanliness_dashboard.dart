import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

// Function to create Cleanliness Pie Chart and Cleanliness Status Line Bars for different platforms
Widget createCleanlinessDashboard(var platform1CleanlinessValue, Color platform1CleanlinessColor, var platform1CleanlinessRadius, var platform2CleanlinessValue, Color platform2CleanlinessColor, var platform2CleanlinessRadius, var platform3CleanlinessValue, Color platform3CleanlinessColor, var platform3CleanlinessRadius, var platform4CleanlinessValue, Color platform4CleanlinessColor, var platform4CleanlinessRadius, var platform5CleanlinessValue, Color platform5CleanlinessColor, var platform5CleanlinessRadius, var platform6CleanlinessValue, Color platform6CleanlinessColor, var platform6CleanlinessRadius){
  return Column(
    children: [
      // Pie Chart
      Stack(
        children: [
          SizedBox(
            height: 250,
            child: PieChart(PieChartData(sections: [
              PieChartSectionData(
                  value: platform1CleanlinessValue + 1,
                  color: platform1CleanlinessColor,
                  showTitle: false,
                  radius: platform1CleanlinessRadius),
              PieChartSectionData(
                  value: platform2CleanlinessValue + 1,
                  color: platform2CleanlinessColor,
                  showTitle: false,
                  radius: platform2CleanlinessRadius),
              PieChartSectionData(
                  value: platform3CleanlinessValue + 1,
                  color: platform3CleanlinessColor,
                  showTitle: false,
                  radius: platform3CleanlinessRadius),
              PieChartSectionData(
                  value: platform4CleanlinessValue + 1,
                  color: platform4CleanlinessColor,
                  showTitle: false,
                  radius: platform4CleanlinessRadius),
              PieChartSectionData(
                  value: platform5CleanlinessValue + 1,
                  color: platform5CleanlinessColor,
                  showTitle: false,
                  radius: platform5CleanlinessRadius),
              PieChartSectionData(
                  value: platform6CleanlinessValue + 1,
                  color: platform6CleanlinessColor,
                  showTitle: false,
                  radius: platform6CleanlinessRadius),
            ])),
          ),

          Positioned.fill(
            child: Align(
                alignment: Alignment.center,
                child: Text("Cleanliness", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: primaryColor),)
            ),
          )
        ],
      ),

      const SizedBox(height: 50),

      // Cleanliness Bars
      Padding(
        padding: const EdgeInsets.all(15.0),
        child: Container(
          decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.black12),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.shade200,
                  blurRadius: 8,
                  spreadRadius: 1,
                )
              ]),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // platform 1 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 1",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform1CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform1CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform1CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform1CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 2 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 2",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform2CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform2CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform2CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform2CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 3 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 3",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform3CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform3CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform3CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform3CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 4 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 4",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform4CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform4CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform4CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform4CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 5 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 5",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform5CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform5CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform5CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform5CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 6 cleanliness percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 6",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform6CleanlinessValue).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform6CleanlinessColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform6CleanlinessValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform6CleanlinessColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),

      const SizedBox(height: 10),
    ],
  );
}


// Function to create pie charts for different status of cleanliness
Widget createCleanlinessTypePieChart(var littleBadStatusValue, Color littleBadColor, var badStatusValue, Color badColor, var worseStatusValue, Color worseColor){
  return Column(
    children: [
      // Cleanliness Status Pie Chart (Row - 1)
      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: littleBadStatusValue + 1,
                          color: littleBadColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - littleBadStatusValue + 1,
                          color: littleBadColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Little Bad - ${double.parse((littleBadStatusValue).toStringAsFixed(2))}%")
              ],
            ),

            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: badStatusValue + 1,
                          color: badColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - badStatusValue + 1,
                          color: badColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Bad - ${double.parse((badStatusValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      // Cleanliness Status Pie Chart (Row - 2)
      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: worseStatusValue + 1,
                          color: worseColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - worseStatusValue + 1,
                          color: worseColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Worse - ${double.parse((worseStatusValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      const SizedBox(height: 50),
    ],
  );
}
