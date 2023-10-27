import 'dart:math';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:liquid_pull_to_refresh/liquid_pull_to_refresh.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/pages/home_widgets/dashboard/cleanliness_dashboard.dart';
import 'package:railway/src/pages/home_widgets/dashboard/crime_dashboard.dart';
import 'package:railway/src/pages/home_widgets/dashboard/crowd_management_dashboard.dart';
import 'package:railway/src/pages/home_widgets/dashboard/work_monitoring_dashboard.dart';
import 'package:railway/src/utils/constants.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // TODO : REQUIRED VARIABLES FOR CRIME DASHBOARD

  // Todo - Required Variables for getCrime() Method
  // variables to store total crimes on a specific platform
  var platform1Crime = 0;
  var platform2Crime = 0;
  var platform3Crime = 0;
  var platform4Crime = 0;
  var platform5Crime = 0;
  var platform6Crime = 0;

  // variables to update radius of pie-chart according to total number of crimes on a platform
  var platform1Radius = 30.0;
  var platform2Radius = 30.0;
  var platform3Radius = 30.0;
  var platform4Radius = 30.0;
  var platform5Radius = 30.0;
  var platform6Radius = 30.0;

  // variable to get platform with maximum and minimum crimes
  var maxCrime = 0;
  var minCrime = 0;

  // variables to store total crimes of all platforms
  var totalCrimes = 0;
  // variable to store multiplier for getting crime percentage
  var multiplier = 0.0;

  // variable to store crime percentage
  var platform1Value = 0.0;
  var platform2Value = 0.0;
  var platform3Value = 0.0;
  var platform4Value = 0.0;
  var platform5Value = 0.0;
  var platform6Value = 0.0;

  // variable to store and update color of pie-chart according to number of crimes
  Color platform1Color = primaryColor;
  Color platform2Color = primaryColor;
  Color platform3Color = primaryColor;
  Color platform4Color = primaryColor;
  Color platform5Color = primaryColor;
  Color platform6Color = primaryColor;

  // Todo - Required Variables for getCrimeType() Method
  // variables to store total number of crime types of all platforms
  var totalArsonClips = 0;
  var totalFightingClips = 0;
  var totalShootingClips = 0;
  var totalNormalClips = 0;
  var totalRobberyClips = 0;

  // variable to store total crimes according to its specific type
  var allCrimes = 0;
  var multiplier2 = 0.0;

  // variable to store crime type percentage
  var arsonValue = 0.0;
  var fightingValue = 0.0;
  var shootingValue = 0.0;
  var normalValue = 0.0;
  var robberyValue = 0.0;

  // variable to get maximum and minimum crime type
  var maxCrimeType = 0;
  var minCrimeType = 0;

  // variable to store and update color of pie-charts of crime type according to number of crimes
  Color arsonColor = primaryColor;
  Color fightingColor = primaryColor;
  Color shootingColor = primaryColor;
  Color normalColor = primaryColor;
  Color robberyColor = primaryColor;
  // ----------X----------X----------X----------X----------X----------

  // TODO : REQUIRED VARIABLES FOR CLEANLINESS DASHBOARD

  // Todo - Required Variables for getCleanliness() Method
  // variables to store total cleanliness on a specific platform
  var platform1Cleanliness = 0;
  var platform2Cleanliness = 0;
  var platform3Cleanliness = 0;
  var platform4Cleanliness = 0;
  var platform5Cleanliness = 0;
  var platform6Cleanliness = 0;

  // variables to update radius of pie-chart according to total number of cleanliness on a platform
  var platform1CleanlinessRadius = 30.0;
  var platform2CleanlinessRadius = 30.0;
  var platform3CleanlinessRadius = 30.0;
  var platform4CleanlinessRadius = 30.0;
  var platform5CleanlinessRadius = 30.0;
  var platform6CleanlinessRadius = 30.0;

  // variable to get platform with maximum and minimum cleanliness
  var maxCleanliness = 0;
  var minCleanliness = 0;

  // variables to store total cleanliness of all platforms
  var totalCleanliness = 0;
  // variable to store multiplier for getting cleanliness percentage
  var cleanlinessMultiplier = 0.0;

  // variable to store cleanliness percentage
  var platform1CleanlinessValue = 0.0;
  var platform2CleanlinessValue = 0.0;
  var platform3CleanlinessValue = 0.0;
  var platform4CleanlinessValue = 0.0;
  var platform5CleanlinessValue = 0.0;
  var platform6CleanlinessValue = 0.0;

  // variable to store and update color of pie-chart according to number of cleanliness
  Color platform1CleanlinessColor = primaryColor;
  Color platform2CleanlinessColor = primaryColor;
  Color platform3CleanlinessColor = primaryColor;
  Color platform4CleanlinessColor = primaryColor;
  Color platform5CleanlinessColor = primaryColor;
  Color platform6CleanlinessColor = primaryColor;

  // Todo - Required Variables for getCleanlinessStatus() Method
  // variables to store total number of cleanliness status of all platforms
  var totalLittleBadStatusClips = 0;
  var totalBadStatusClips = 0;
  var totalWorseStatusClips = 0;

  // variable to store total cleanliness according to its specific type
  var allCleanliness = 0;
  var cleanlinessMultiplier2 = 0.0;

  // variable to store cleanliness type percentage
  var littleBadStatusValue = 0.0;
  var badStatusValue = 0.0;
  var worseStatusValue = 0.0;

  // variable to store color of pie-charts according to cleanliness status
  Color littleBadColor = Colors.yellow;
  Color badColor = Colors.orange;
  Color worseColor = Colors.red;
  // ----------X----------X----------X----------X----------X----------

  // TODO : REQUIRED VARIABLES FOR WORK MONITORING DASHBOARD

  // Todo - Required Variables for getWorkMonitoring() Method
  // variables to store total work monitoring on a specific platform
  var platform1WorkMonitoring = 0;
  var platform2WorkMonitoring = 0;
  var platform3WorkMonitoring = 0;
  var platform4WorkMonitoring = 0;
  var platform5WorkMonitoring = 0;
  var platform6WorkMonitoring = 0;

  // variables to update radius of pie-chart according to total number of work monitoring on a platform
  var platform1WorkMonitoringRadius = 30.0;
  var platform2WorkMonitoringRadius = 30.0;
  var platform3WorkMonitoringRadius = 30.0;
  var platform4WorkMonitoringRadius = 30.0;
  var platform5WorkMonitoringRadius = 30.0;
  var platform6WorkMonitoringRadius = 30.0;

  // variable to get platform with maximum and minimum work monitoring
  var maxWorkMonitoring = 0;
  var minWorkMonitoring = 0;

  // variables to store total work monitoring of all platforms
  var totalWorkMonitoring = 0;
  // variable to store multiplier for getting work monitoring percentage
  var workMonitoringMultiplier = 0.0;

  // variable to store work monitoring percentage
  var platform1WorkMonitoringValue = 0.0;
  var platform2WorkMonitoringValue = 0.0;
  var platform3WorkMonitoringValue = 0.0;
  var platform4WorkMonitoringValue = 0.0;
  var platform5WorkMonitoringValue = 0.0;
  var platform6WorkMonitoringValue = 0.0;

  // variable to store and update color of pie-chart according to number of work monitoring
  Color platform1WorkMonitoringColor = primaryColor;
  Color platform2WorkMonitoringColor = primaryColor;
  Color platform3WorkMonitoringColor = primaryColor;
  Color platform4WorkMonitoringColor = primaryColor;
  Color platform5WorkMonitoringColor = primaryColor;
  Color platform6WorkMonitoringColor = primaryColor;
  // ----------X----------X----------X----------X----------X----------

  // TODO : REQUIRED VARIABLES FOR CROWD MANAGEMENT DASHBOARD

  // Todo - Required Variables for getCrowd() Method
  // variables to store total crowd detection on a specific platform
  var platform1CrowdManagement = 0;
  var platform2CrowdManagement = 0;
  var platform3CrowdManagement = 0;
  var platform4CrowdManagement = 0;
  var platform5CrowdManagement = 0;
  var platform6CrowdManagement = 0;

  // variables to update radius of pie-chart according to total number of crowd detection on a platform
  var platform1CrowdManagementRadius = 30.0;
  var platform2CrowdManagementRadius = 30.0;
  var platform3CrowdManagementRadius = 30.0;
  var platform4CrowdManagementRadius = 30.0;
  var platform5CrowdManagementRadius = 30.0;
  var platform6CrowdManagementRadius = 30.0;

  // variable to get platform with maximum and minimum crowd detection
  var maxCrowdManagement = 0;
  var minCrowdManagement = 0;

  // variables to store total crowd detection of all platforms
  var totalCrowdManagement = 0;
  // variable to store multiplier for getting crowd detection percentage
  var crowdManagementMultiplier = 0.0;

  // variable to store crowd detection percentage
  var platform1CrowdManagementValue = 0.0;
  var platform2CrowdManagementValue = 0.0;
  var platform3CrowdManagementValue = 0.0;
  var platform4CrowdManagementValue = 0.0;
  var platform5CrowdManagementValue = 0.0;
  var platform6CrowdManagementValue = 0.0;

  // variable to store and update color of pie-chart according to number of crowd detection
  Color platform1CrowdManagementColor = primaryColor;
  Color platform2CrowdManagementColor = primaryColor;
  Color platform3CrowdManagementColor = primaryColor;
  Color platform4CrowdManagementColor = primaryColor;
  Color platform5CrowdManagementColor = primaryColor;
  Color platform6CrowdManagementColor = primaryColor;

  // Todo - Required Variables for getCrowdManagementStatus() Method
  // variables to store total number of crowd detection status of all platforms
  var totalLessCrowdClips = 0;
  var totalNormalCrowdClips = 0;
  var totalHighCrowdClips = 0;
  var totalVeryHighCrowdClips = 0;

  // variable to store total crowd detection according to its specific type
  var allCrowdManagement = 0;
  var crowdManagementMultiplier2 = 0.0;

  // variable to store crowd detection status percentage
  var lessCrowdValue = 0.0;
  var normalCrowdValue = 0.0;
  var highCrowdValue = 0.0;
  var veryHighCrowdValue = 0.0;

  // variable to store color of pie-charts according to crowd status
  Color lessCrowdColor = Colors.green;
  Color normalCrowdColor = Colors.yellow;
  Color highCrowdColor = Colors.orange;
  Color veryHighCrowdColor = Colors.red;
  // ----------X----------X----------X----------X----------X----------

  // TODO : REQUIRED VARIABLES FOR BUTTON OF THE TOP ROW
  String title = "CRIME";
  Color crimeButtonBackgroundColor = primaryColor;
  Color crimeButtonTextColor = Colors.white;

  Color cleanlinessButtonBackgroundColor = Colors.white;
  Color cleanlinessButtonTextColor = primaryColor;

  Color workMonitoringButtonBackgroundColor = Colors.white;
  Color workMonitoringButtonTextColor = primaryColor;

  Color crowdManagementButtonBackgroundColor = Colors.white;
  Color crowdManagementButtonTextColor = primaryColor;
  // ----------X----------X----------X----------X----------X----------

  // TODO : REQUIRED FUNCTIONS FOR DIFFERENT DASHBOARDS
  // function to create section heading
  Widget sectionHeading(String text) {
    return Padding(
      padding: const EdgeInsets.only(left: 10.0, top: 10.0, bottom: 20.0),
      child: Align(
        alignment: Alignment.topLeft,
        child: Text(
          text,
          style: const TextStyle(fontSize: 14, color: Colors.black54),
        ),
      ),
    );
  }
  // ----------X----------X----------X----------X----------X----------

  // Todo - Function to create crime dashboard
  // function to get Crime Type for pie chart of specific crime type
  void getCrimeType() async {
    var query1 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('label', isEqualTo: "Arson");
    var snapshot1 = await query1.get();
    totalArsonClips = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('label', isEqualTo: "Fighting");
    var snapshot2 = await query2.get();
    totalFightingClips = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('label', isEqualTo: "Shooting");
    var snapshot3 = await query3.get();
    totalShootingClips = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('label', isEqualTo: "Normal");
    var snapshot4 = await query4.get();
    totalNormalClips = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('label', isEqualTo: "Robbery");
    var snapshot5 = await query5.get();
    totalRobberyClips = snapshot5.size;

    maxCrimeType = [
      totalArsonClips,
      totalFightingClips,
      totalShootingClips,
      totalNormalClips,
      totalRobberyClips
    ].reduce(max);
    minCrimeType = [
      totalArsonClips,
      totalFightingClips,
      totalShootingClips,
      totalNormalClips,
      totalRobberyClips
    ].reduce(min);

    allCrimes = totalArsonClips +
        totalFightingClips +
        totalShootingClips +
        totalNormalClips +
        totalRobberyClips;
    multiplier2 = 100 / allCrimes;

    arsonValue = totalArsonClips * multiplier2;
    fightingValue = totalFightingClips * multiplier2;
    shootingValue = totalShootingClips * multiplier2;
    normalValue = totalNormalClips * multiplier2;
    robberyValue = totalRobberyClips * multiplier2;

    // For Arson Crime
    if (totalArsonClips == maxCrimeType) {
      arsonColor = Colors.red;
    } else if (totalArsonClips == minCrimeType) {
      arsonColor = Colors.green;
    } else {
      arsonColor = Colors.orange;
    }

    // For Fighting Crime
    if (totalFightingClips == maxCrimeType) {
      fightingColor = Colors.red;
    } else if (totalFightingClips == minCrimeType) {
      fightingColor = Colors.green;
    } else {
      fightingColor = Colors.orange;
    }

    // For Shooting Crime
    if (totalShootingClips == maxCrimeType) {
      shootingColor = Colors.red;
    } else if (totalShootingClips == minCrimeType) {
      shootingColor = Colors.green;
    } else {
      shootingColor = Colors.orange;
    }

    // For Normal Crime
    if (totalNormalClips == maxCrimeType) {
      normalColor = Colors.red;
    } else if (totalNormalClips == minCrimeType) {
      normalColor = Colors.green;
    } else {
      normalColor = Colors.orange;
    }

    // For Robbery Crime
    if (totalRobberyClips == maxCrimeType) {
      robberyColor = Colors.red;
    } else if (totalRobberyClips == minCrimeType) {
      robberyColor = Colors.green;
    } else {
      robberyColor = Colors.orange;
    }
  }

  // function to get Crimes for Pie chart and Platform specific Crime bar
  void getCrimes() async {
    var query1 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1Crime = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2Crime = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3Crime = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4Crime = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5Crime = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6Crime = snapshot6.size;

    maxCrime = [
      platform1Crime,
      platform2Crime,
      platform3Crime,
      platform4Crime,
      platform5Crime,
      platform6Crime
    ].reduce(max);
    minCrime = [
      platform1Crime,
      platform2Crime,
      platform3Crime,
      platform4Crime,
      platform5Crime,
      platform6Crime
    ].reduce(min);

    totalCrimes = platform1Crime +
        platform2Crime +
        platform3Crime +
        platform4Crime +
        platform5Crime +
        platform6Crime;
    multiplier = 100 / totalCrimes;

    platform1Value = platform1Crime * multiplier;
    platform2Value = platform2Crime * multiplier;
    platform3Value = platform3Crime * multiplier;
    platform4Value = platform4Crime * multiplier;
    platform5Value = platform5Crime * multiplier;
    platform6Value = platform6Crime * multiplier;

    if (kDebugMode) {
      print("Platform 1 Value = ${platform1Value / 100}");
      print("Platform 2 Value = ${platform2Value / 100}");
      print("Platform 3 Value = ${platform3Value / 100}");
      print("Platform 4 Value = ${platform4Value / 100}");
      print("Platform 5 Value = ${platform5Value / 100}");
      print("Platform 6 Value = ${platform6Value / 100}");
    }

    // For Platform 1 Crime
    if (platform1Crime == maxCrime) {
      platform1Radius = 40;
      platform1Color = Colors.red;
    } else if (platform1Crime == minCrime) {
      platform1Radius = 30;
      platform1Color = Colors.green;
    } else {
      platform1Radius = 35;
      platform1Color = Colors.orange;
    }

    // For Platform 2 Crime
    if (platform2Crime == maxCrime) {
      platform2Radius = 40;
      platform2Color = Colors.red;
    } else if (platform2Crime == minCrime) {
      platform2Radius = 30;
      platform2Color = Colors.green;
    } else {
      platform2Radius = 35;
      platform2Color = Colors.orange;
    }

    // For Platform 3 Crime
    if (platform3Crime == maxCrime) {
      platform3Radius = 40;
      platform3Color = Colors.red;
    } else if (platform3Crime == minCrime) {
      platform3Radius = 30;
      platform3Color = Colors.green;
    } else {
      platform3Radius = 35;
      platform3Color = Colors.orange;
    }

    // For Platform 4 Crime
    if (platform4Crime == maxCrime) {
      platform4Radius = 40;
      platform4Color = Colors.red;
    } else if (platform4Crime == minCrime) {
      platform4Radius = 30;
      platform4Color = Colors.green;
    } else {
      platform4Radius = 35;
      platform4Color = Colors.orange;
    }

    // For Platform 5 Crime
    if (platform5Crime == maxCrime) {
      platform5Radius = 40;
      platform5Color = Colors.red;
    } else if (platform5Crime == minCrime) {
      platform5Radius = 30;
      platform5Color = Colors.green;
    } else {
      platform5Radius = 35;
      platform5Color = Colors.orange;
    }

    // For Platform 6 Crime
    if (platform6Crime == maxCrime) {
      platform6Radius = 40;
      platform6Color = Colors.red;
    } else if (platform6Crime == minCrime) {
      platform6Radius = 30;
      platform6Color = Colors.green;
    } else {
      platform6Radius = 35;
      platform6Color = Colors.orange;
    }
  }
  // ----------X----------X----------X----------X----------X----------

  // Todo - Function to create cleanliness dashboard
  // function to get Crime Type for pie chart of specific crime type
  void getCleanlinessStatus() async {
    var query1 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('status', isEqualTo: "Little Bad");
    var snapshot1 = await query1.get();
    totalLittleBadStatusClips = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('status', isEqualTo: "Bad");
    var snapshot2 = await query2.get();
    totalBadStatusClips = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('status', isEqualTo: "Worse");
    var snapshot3 = await query3.get();
    totalWorseStatusClips = snapshot3.size;

    allCleanliness =
        totalLittleBadStatusClips + totalBadStatusClips + totalWorseStatusClips;
    cleanlinessMultiplier2 = 100 / allCleanliness;

    littleBadStatusValue = totalLittleBadStatusClips * cleanlinessMultiplier2;
    badStatusValue = totalBadStatusClips * cleanlinessMultiplier2;
    worseStatusValue = totalWorseStatusClips * cleanlinessMultiplier2;
  }

  // function to get Crimes for Pie chart and Platform specific Crime bar
  void getCleanliness() async {
    var query1 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1Cleanliness = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2Cleanliness = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3Cleanliness = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4Cleanliness = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5Cleanliness = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6Cleanliness = snapshot6.size;

    maxCleanliness = [
      platform1Cleanliness,
      platform2Cleanliness,
      platform3Cleanliness,
      platform4Cleanliness,
      platform5Cleanliness,
      platform6Cleanliness
    ].reduce(max);

    minCleanliness = [
      platform1Cleanliness,
      platform2Cleanliness,
      platform3Cleanliness,
      platform4Cleanliness,
      platform5Cleanliness,
      platform6Cleanliness
    ].reduce(min);

    totalCleanliness = platform1Cleanliness +
        platform2Cleanliness +
        platform3Cleanliness +
        platform4Cleanliness +
        platform5Cleanliness +
        platform6Cleanliness;
    cleanlinessMultiplier = 100 / totalCleanliness;

    platform1CleanlinessValue = platform1Cleanliness * cleanlinessMultiplier;
    platform2CleanlinessValue = platform2Cleanliness * cleanlinessMultiplier;
    platform3CleanlinessValue = platform3Cleanliness * cleanlinessMultiplier;
    platform4CleanlinessValue = platform4Cleanliness * cleanlinessMultiplier;
    platform5CleanlinessValue = platform5Cleanliness * cleanlinessMultiplier;
    platform6CleanlinessValue = platform6Cleanliness * cleanlinessMultiplier;

    if (kDebugMode) {
      print(
          "Platform 1 Cleanliness Value = ${platform1CleanlinessValue / 100}");
      print(
          "Platform 2 Cleanliness Value = ${platform2CleanlinessValue / 100}");
      print(
          "Platform 3 Cleanliness Value = ${platform3CleanlinessValue / 100}");
      print(
          "Platform 4 Cleanliness Value = ${platform4CleanlinessValue / 100}");
      print(
          "Platform 5 Cleanliness Value = ${platform5CleanlinessValue / 100}");
      print(
          "Platform 6 Cleanliness Value = ${platform6CleanlinessValue / 100}");
    }

    // For Platform 1 Crime
    if (platform1Cleanliness == maxCleanliness) {
      platform1CleanlinessRadius = 40;
      platform1CleanlinessColor = Colors.red;
    } else if (platform1Cleanliness == minCleanliness) {
      platform1CleanlinessRadius = 30;
      platform1CleanlinessColor = Colors.green;
    } else {
      platform1CleanlinessRadius = 35;
      platform1CleanlinessColor = Colors.orange;
    }

    // For Platform 2 Crime
    if (platform2Cleanliness == maxCleanliness) {
      platform2CleanlinessRadius = 40;
      platform2CleanlinessColor = Colors.red;
    } else if (platform2Cleanliness == minCleanliness) {
      platform2CleanlinessRadius = 30;
      platform2CleanlinessColor = Colors.green;
    } else {
      platform2CleanlinessRadius = 35;
      platform2CleanlinessColor = Colors.orange;
    }

    // For Platform 3 Crime
    if (platform3Cleanliness == maxCleanliness) {
      platform3CleanlinessRadius = 40;
      platform3CleanlinessColor = Colors.red;
    } else if (platform3Cleanliness == minCleanliness) {
      platform3CleanlinessRadius = 30;
      platform3CleanlinessColor = Colors.green;
    } else {
      platform3CleanlinessRadius = 35;
      platform3CleanlinessColor = Colors.orange;
    }

    // For Platform 4 Crime
    if (platform4Cleanliness == maxCleanliness) {
      platform4CleanlinessRadius = 40;
      platform4CleanlinessColor = Colors.red;
    } else if (platform4Cleanliness == minCleanliness) {
      platform4CleanlinessRadius = 30;
      platform4CleanlinessColor = Colors.green;
    } else {
      platform4CleanlinessRadius = 35;
      platform4CleanlinessColor = Colors.orange;
    }

    // For Platform 5 Crime
    if (platform5Cleanliness == maxCleanliness) {
      platform5CleanlinessRadius = 40;
      platform5CleanlinessColor = Colors.red;
    } else if (platform5Cleanliness == minCleanliness) {
      platform5CleanlinessRadius = 30;
      platform5CleanlinessColor = Colors.green;
    } else {
      platform5CleanlinessRadius = 35;
      platform5CleanlinessColor = Colors.orange;
    }

    // For Platform 6 Crime
    if (platform6Cleanliness == maxCleanliness) {
      platform6CleanlinessRadius = 40;
      platform6CleanlinessColor = Colors.red;
    } else if (platform6Cleanliness == minCleanliness) {
      platform6CleanlinessRadius = 30;
      platform6CleanlinessColor = Colors.green;
    } else {
      platform6CleanlinessRadius = 35;
      platform6CleanlinessColor = Colors.orange;
    }
  }
  // ----------X----------X----------X----------X----------X----------

  // Todo - Function to create work monitoring dashboard
  // function to get Crimes for Pie chart and Platform specific Crime bar
  void getWorkMonitoring() async {
    var query1 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1WorkMonitoring = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2WorkMonitoring = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3WorkMonitoring = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4WorkMonitoring = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5WorkMonitoring = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6WorkMonitoring = snapshot6.size;

    maxWorkMonitoring = [
      platform1WorkMonitoring,
      platform2WorkMonitoring,
      platform3WorkMonitoring,
      platform4WorkMonitoring,
      platform5WorkMonitoring,
      platform6WorkMonitoring
    ].reduce(max);

    minWorkMonitoring = [
      platform1WorkMonitoring,
      platform2WorkMonitoring,
      platform3WorkMonitoring,
      platform4WorkMonitoring,
      platform5WorkMonitoring,
      platform6WorkMonitoring
    ].reduce(min);

    totalWorkMonitoring = platform1WorkMonitoring +
        platform2WorkMonitoring +
        platform3WorkMonitoring +
        platform4WorkMonitoring +
        platform5WorkMonitoring +
        platform6WorkMonitoring;
    workMonitoringMultiplier = 100 / totalWorkMonitoring;

    platform1WorkMonitoringValue =
        platform1WorkMonitoring * workMonitoringMultiplier;
    platform2WorkMonitoringValue =
        platform2WorkMonitoring * workMonitoringMultiplier;
    platform3WorkMonitoringValue =
        platform3WorkMonitoring * workMonitoringMultiplier;
    platform4WorkMonitoringValue =
        platform4WorkMonitoring * workMonitoringMultiplier;
    platform5WorkMonitoringValue =
        platform5WorkMonitoring * workMonitoringMultiplier;
    platform6WorkMonitoringValue =
        platform6WorkMonitoring * workMonitoringMultiplier;

    if (kDebugMode) {
      print(
          "Platform 1 Work Monitoring Value = ${platform1WorkMonitoringValue / 100}");
      print(
          "Platform 2 Work Monitoring Value = ${platform2WorkMonitoringValue / 100}");
      print(
          "Platform 3 Work Monitoring Value = ${platform3WorkMonitoringValue / 100}");
      print(
          "Platform 4 Work Monitoring Value = ${platform4WorkMonitoringValue / 100}");
      print(
          "Platform 5 Work Monitoring Value = ${platform5WorkMonitoringValue / 100}");
      print(
          "Platform 6 Work Monitoring Value = ${platform6WorkMonitoringValue / 100}");
    }

    // For Platform 1 Crime
    if (platform1WorkMonitoring == maxWorkMonitoring) {
      platform1WorkMonitoringRadius = 40;
      platform1WorkMonitoringColor = Colors.red;
    } else if (platform1WorkMonitoring == minWorkMonitoring) {
      platform1WorkMonitoringRadius = 30;
      platform1WorkMonitoringColor = Colors.green;
    } else {
      platform1WorkMonitoringRadius = 35;
      platform1WorkMonitoringColor = Colors.orange;
    }

    // For Platform 2 Crime
    if (platform2WorkMonitoring == maxWorkMonitoring) {
      platform2WorkMonitoringRadius = 40;
      platform2WorkMonitoringColor = Colors.red;
    } else if (platform2WorkMonitoring == minWorkMonitoring) {
      platform2WorkMonitoringRadius = 30;
      platform2WorkMonitoringColor = Colors.green;
    } else {
      platform2WorkMonitoringRadius = 35;
      platform2WorkMonitoringColor = Colors.orange;
    }

    // For Platform 3 Crime
    if (platform3WorkMonitoring == maxWorkMonitoring) {
      platform3WorkMonitoringRadius = 40;
      platform3WorkMonitoringColor = Colors.red;
    } else if (platform3WorkMonitoring == minWorkMonitoring) {
      platform3WorkMonitoringRadius = 30;
      platform3WorkMonitoringColor = Colors.green;
    } else {
      platform3WorkMonitoringRadius = 35;
      platform3WorkMonitoringColor = Colors.orange;
    }

    // For Platform 4 Crime
    if (platform4WorkMonitoring == maxWorkMonitoring) {
      platform4WorkMonitoringRadius = 40;
      platform4WorkMonitoringColor = Colors.red;
    } else if (platform4WorkMonitoring == minWorkMonitoring) {
      platform4WorkMonitoringRadius = 30;
      platform4WorkMonitoringColor = Colors.green;
    } else {
      platform4WorkMonitoringRadius = 35;
      platform4WorkMonitoringColor = Colors.orange;
    }

    // For Platform 5 Crime
    if (platform5WorkMonitoring == maxWorkMonitoring) {
      platform5WorkMonitoringRadius = 40;
      platform5WorkMonitoringColor = Colors.red;
    } else if (platform5WorkMonitoring == minWorkMonitoring) {
      platform5WorkMonitoringRadius = 30;
      platform5WorkMonitoringColor = Colors.green;
    } else {
      platform5WorkMonitoringRadius = 35;
      platform5WorkMonitoringColor = Colors.orange;
    }

    // For Platform 6 Crime
    if (platform6WorkMonitoring == maxWorkMonitoring) {
      platform6WorkMonitoringRadius = 40;
      platform6WorkMonitoringColor = Colors.red;
    } else if (platform6WorkMonitoring == minWorkMonitoring) {
      platform6WorkMonitoringRadius = 30;
      platform6WorkMonitoringColor = Colors.green;
    } else {
      platform6WorkMonitoringRadius = 35;
      platform6WorkMonitoringColor = Colors.orange;
    }
  }
  // ----------X----------X----------X----------X----------X----------

  // Todo - Function to create crowd management dashboard
  // function to get Crime Type for pie chart of specific crime type
  void getCrowdManagementStatus() async {
    var query1 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('label', isEqualTo: "Less Crowd");
    var snapshot1 = await query1.get();
    totalLessCrowdClips = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('label', isEqualTo: "Normal Crowd");
    var snapshot2 = await query2.get();
    totalNormalCrowdClips = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('label', isEqualTo: "High Crowd");
    var snapshot3 = await query3.get();
    totalHighCrowdClips = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('label', isEqualTo: "Very High Crowd");
    var snapshot4 = await query4.get();
    totalVeryHighCrowdClips = snapshot4.size;

    allCrowdManagement = totalLessCrowdClips +
        totalNormalCrowdClips +
        totalHighCrowdClips +
        totalVeryHighCrowdClips;
    crowdManagementMultiplier2 = 100 / allCrowdManagement;

    lessCrowdValue = totalLessCrowdClips * crowdManagementMultiplier2;
    normalCrowdValue = totalNormalCrowdClips * crowdManagementMultiplier2;
    highCrowdValue = totalHighCrowdClips * crowdManagementMultiplier2;
    veryHighCrowdValue = totalVeryHighCrowdClips * crowdManagementMultiplier2;
  }

  // function to get Crimes for Pie chart and Platform specific Crime bar
  void getCrowd() async {
    var query1 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1CrowdManagement = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2CrowdManagement = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3CrowdManagement = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4CrowdManagement = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5CrowdManagement = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6CrowdManagement = snapshot6.size;

    maxCrowdManagement = [
      platform1CrowdManagement,
      platform2CrowdManagement,
      platform3CrowdManagement,
      platform4CrowdManagement,
      platform5CrowdManagement,
      platform6CrowdManagement
    ].reduce(max);

    minCrowdManagement = [
      platform1CrowdManagement,
      platform2CrowdManagement,
      platform3CrowdManagement,
      platform4CrowdManagement,
      platform5CrowdManagement,
      platform6CrowdManagement
    ].reduce(min);

    totalCrowdManagement = platform1CrowdManagement +
        platform2CrowdManagement +
        platform3CrowdManagement +
        platform4CrowdManagement +
        platform5CrowdManagement +
        platform6CrowdManagement;
    crowdManagementMultiplier = 100 / totalCrowdManagement;

    platform1CrowdManagementValue =
        platform1CrowdManagement * crowdManagementMultiplier;
    platform2CrowdManagementValue =
        platform2CrowdManagement * crowdManagementMultiplier;
    platform3CrowdManagementValue =
        platform3CrowdManagement * crowdManagementMultiplier;
    platform4CrowdManagementValue =
        platform4CrowdManagement * crowdManagementMultiplier;
    platform5CrowdManagementValue =
        platform5CrowdManagement * crowdManagementMultiplier;
    platform6CrowdManagementValue =
        platform6CrowdManagement * crowdManagementMultiplier;

    if (kDebugMode) {
      print(
          "Platform 1 Crowd Management Value = ${platform1CrowdManagementValue / 100}");
      print(
          "Platform 2 Crowd Management Value = ${platform2CrowdManagementValue / 100}");
      print(
          "Platform 3 Crowd Management Value = ${platform3CrowdManagementValue / 100}");
      print(
          "Platform 4 Crowd Management Value = ${platform4CrowdManagementValue / 100}");
      print(
          "Platform 5 Crowd Management Value = ${platform5CrowdManagementValue / 100}");
      print(
          "Platform 6 Crowd Management Value = ${platform6CrowdManagementValue / 100}");
    }

    // For Platform 1 Crime
    if (platform1CrowdManagement == maxCrowdManagement) {
      platform1CrowdManagementRadius = 40;
      platform1CrowdManagementColor = Colors.red;
    } else if (platform1CrowdManagement == minCrowdManagement) {
      platform1CrowdManagementRadius = 30;
      platform1CrowdManagementColor = Colors.green;
    } else {
      platform1CrowdManagementRadius = 35;
      platform1CrowdManagementColor = Colors.orange;
    }

    // For Platform 2 Crime
    if (platform2CrowdManagement == maxCrowdManagement) {
      platform2CrowdManagementRadius = 40;
      platform2CrowdManagementColor = Colors.red;
    } else if (platform2CrowdManagement == minCrowdManagement) {
      platform2CrowdManagementRadius = 30;
      platform2CrowdManagementColor = Colors.green;
    } else {
      platform2CrowdManagementRadius = 35;
      platform2CrowdManagementColor = Colors.orange;
    }

    // For Platform 3 Crime
    if (platform3CrowdManagement == maxCrowdManagement) {
      platform3CrowdManagementRadius = 40;
      platform3CrowdManagementColor = Colors.red;
    } else if (platform3CrowdManagement == minCrowdManagement) {
      platform3CrowdManagementRadius = 30;
      platform3CrowdManagementColor = Colors.green;
    } else {
      platform3CrowdManagementRadius = 35;
      platform3CrowdManagementColor = Colors.orange;
    }

    // For Platform 4 Crime
    if (platform4CrowdManagement == maxCrowdManagement) {
      platform4CrowdManagementRadius = 40;
      platform4CrowdManagementColor = Colors.red;
    } else if (platform4CrowdManagement == minCrowdManagement) {
      platform4CrowdManagementRadius = 30;
      platform4CrowdManagementColor = Colors.green;
    } else {
      platform4CrowdManagementRadius = 35;
      platform4CrowdManagementColor = Colors.orange;
    }

    // For Platform 5 Crime
    if (platform5CrowdManagement == maxCrowdManagement) {
      platform5CrowdManagementRadius = 40;
      platform5CrowdManagementColor = Colors.red;
    } else if (platform5CrowdManagement == minCrowdManagement) {
      platform5CrowdManagementRadius = 30;
      platform5CrowdManagementColor = Colors.green;
    } else {
      platform5CrowdManagementRadius = 35;
      platform5CrowdManagementColor = Colors.orange;
    }

    // For Platform 6 Crime
    if (platform6CrowdManagement == maxCrowdManagement) {
      platform6CrowdManagementRadius = 40;
      platform6CrowdManagementColor = Colors.red;
    } else if (platform6CrowdManagement == minCrowdManagement) {
      platform6CrowdManagementRadius = 30;
      platform6CrowdManagementColor = Colors.green;
    } else {
      platform6CrowdManagementRadius = 35;
      platform6CrowdManagementColor = Colors.orange;
    }
  }
  // ----------X----------X----------X----------X----------X----------

  // TODO : METHOD TO CALL ALL METHOD FOR REFRESHING THE PAGE
  // function to reload dashboard
  Future<void> _handleRefresh() async {
    setState(() {
      getCrimes();
      getCrimeType();

      getCleanliness();
      getCleanlinessStatus();

      getWorkMonitoring();

      getCrowdManagementStatus();
      getCrowd();
    });
    return await Future.delayed(const Duration(seconds: 3));
  }

  // Initial method
  @override
  void initState() {
    super.initState();

    setState(() {
      getCrimes();
      getCrimeType();

      getCleanliness();
      getCleanlinessStatus();

      getWorkMonitoring();

      getCrowdManagementStatus();
      getCrowd();
    });
  }
  // ----------X----------X----------X----------X----------X----------

  // TODO - BODY
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LiquidPullToRefresh(
        onRefresh: _handleRefresh,
        color: Colors.transparent,
        backgroundColor: primaryColor,
        height: 300,
        animSpeedFactor: 3,
        showChildOpacityTransition: true,
        child: ListView(
          physics: const BouncingScrollPhysics(),
          children: [
            // Title
            Padding(
              padding: const EdgeInsets.only(
                  left: 20.0, right: 8.0, bottom: 15.0, top: 20.0),
              child: Stack(
                children: [
                  Align(
                      alignment: Alignment.topLeft,
                      child: GestureDetector(
                          onTap: () {
                            Navigator.pop(context);
                          },
                          child:
                              Icon(Icons.arrow_back_ios, color: primaryColor))),
                  Align(
                    alignment: Alignment.center,
                    child: Center(
                        child: Text(
                      "$title DASHBOARD",
                      style: TextStyle(
                          color: primaryColor,
                          fontWeight: FontWeight.bold,
                          fontSize: 18),
                    )),
                  ),
                ],
              ),
            ),

            // Button Row
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  // Button 1
                  Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      decoration: BoxDecoration(
                          color: crimeButtonBackgroundColor,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.black12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.shade200,
                              blurRadius: 10,
                              spreadRadius: 1,
                            )
                          ]),
                      child: TextButton(
                        onPressed: () {
                          setState(() {
                            title = "CRIME";

                            crimeButtonBackgroundColor = primaryColor;
                            crimeButtonTextColor = Colors.white;

                            cleanlinessButtonBackgroundColor = Colors.white;
                            cleanlinessButtonTextColor = primaryColor;

                            workMonitoringButtonBackgroundColor = Colors.white;
                            workMonitoringButtonTextColor = primaryColor;

                            crowdManagementButtonBackgroundColor = Colors.white;
                            crowdManagementButtonTextColor = primaryColor;
                          });
                        },
                        child: Text(
                          "Crime",
                          style: TextStyle(
                              color: crimeButtonTextColor, fontSize: 18),
                        ),
                      ),
                    ),
                  ),

                  // Button 2
                  Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      decoration: BoxDecoration(
                          color: cleanlinessButtonBackgroundColor,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.black12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.shade200,
                              blurRadius: 10,
                              spreadRadius: 1,
                            )
                          ]),
                      child: TextButton(
                        onPressed: () {
                          setState(() {
                            title = "CLEANLINESS";

                            crimeButtonBackgroundColor = Colors.white;
                            crimeButtonTextColor = primaryColor;

                            cleanlinessButtonBackgroundColor = primaryColor;
                            cleanlinessButtonTextColor = Colors.white;

                            workMonitoringButtonBackgroundColor = Colors.white;
                            workMonitoringButtonTextColor = primaryColor;

                            crowdManagementButtonBackgroundColor = Colors.white;
                            crowdManagementButtonTextColor = primaryColor;
                          });
                        },
                        child: Text(
                          "Cleanliness",
                          style: TextStyle(
                              color: cleanlinessButtonTextColor, fontSize: 18),
                        ),
                      ),
                    ),
                  ),

                  // Button 3
                  Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      decoration: BoxDecoration(
                          color: workMonitoringButtonBackgroundColor,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.black12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.shade200,
                              blurRadius: 10,
                              spreadRadius: 1,
                            )
                          ]),
                      child: TextButton(
                        onPressed: () {
                          setState(() {
                            title = "WORK MONITORING";

                            crimeButtonBackgroundColor = Colors.white;
                            crimeButtonTextColor = primaryColor;

                            cleanlinessButtonBackgroundColor = Colors.white;
                            cleanlinessButtonTextColor = primaryColor;

                            workMonitoringButtonBackgroundColor = primaryColor;
                            workMonitoringButtonTextColor = Colors.white;

                            crowdManagementButtonBackgroundColor = Colors.white;
                            crowdManagementButtonTextColor = primaryColor;
                          });
                        },
                        child: Text(
                          "Work Monitoring",
                          style: TextStyle(
                              color: workMonitoringButtonTextColor,
                              fontSize: 18),
                        ),
                      ),
                    ),
                  ),

                  // Button 4
                  Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      decoration: BoxDecoration(
                          color: crowdManagementButtonBackgroundColor,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.black12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.shade200,
                              blurRadius: 10,
                              spreadRadius: 1,
                            )
                          ]),
                      child: TextButton(
                        onPressed: () {
                          setState(() {
                            title = "CROWD MANAGEMENT";

                            crimeButtonBackgroundColor = Colors.white;
                            crimeButtonTextColor = primaryColor;

                            cleanlinessButtonBackgroundColor = Colors.white;
                            cleanlinessButtonTextColor = primaryColor;

                            workMonitoringButtonBackgroundColor = Colors.white;
                            workMonitoringButtonTextColor = primaryColor;

                            crowdManagementButtonBackgroundColor = primaryColor;
                            crowdManagementButtonTextColor = Colors.white;
                          });
                        },
                        child: Text(
                          "Crowd Management",
                          style: TextStyle(
                              color: crowdManagementButtonTextColor,
                              fontSize: 18),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Crime Dashboard
            title.toLowerCase() == "crime"
                ? totalCrimes == 0
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.start,
                        children: [
                          const SizedBox(
                            height: 50,
                          ),
                          Center(
                            child: Lottie.asset(
                                "assets/animations/no_data_available.json",
                                animate: true,
                                width: 350),
                          ),
                          Center(
                            child: Text(
                              title,
                              style: TextStyle(
                                  fontSize: 24,
                                  color: primaryColor,
                                  fontWeight: FontWeight.bold),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Center(
                              child: Text(
                                "No Data Found!",
                                style: TextStyle(
                                    fontSize: 20, color: secondaryColor),
                              ),
                            ),
                          ),
                        ],
                      )
                    : Column(
                        children: [
                          sectionHeading("Platform Based Crimes"),

                          // Creating Crime Pie Chart and Crime Line Bars for different platforms
                          createCrimeDashboard(
                              platform1Value,
                              platform1Color,
                              platform1Radius,
                              platform2Value,
                              platform2Color,
                              platform2Radius,
                              platform3Value,
                              platform3Color,
                              platform3Radius,
                              platform4Value,
                              platform4Color,
                              platform4Radius,
                              platform5Value,
                              platform5Color,
                              platform5Radius,
                              platform6Value,
                              platform6Color,
                              platform6Radius),

                          sectionHeading("Crime Types"),
                          // Creating pie charts for different types of crime
                          createCrimeTypePieChart(
                              arsonValue,
                              arsonColor,
                              fightingValue,
                              fightingColor,
                              shootingValue,
                              shootingColor,
                              robberyValue,
                              robberyColor,
                              normalValue,
                              normalColor)
                        ],
                      )
                : title.toLowerCase() == "cleanliness"
                    ? totalCleanliness == 0
                        ? Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                const SizedBox(
                  height: 50,
                ),
                Center(
                  child: Lottie.asset(
                      "assets/animations/no_data_available.json",
                      animate: true,
                      width: 350),
                ),
                Center(
                  child: Text(
                    title,
                    style: TextStyle(
                        fontSize: 24,
                        color: primaryColor,
                        fontWeight: FontWeight.bold),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Center(
                    child: Text(
                      "No Data Found!",
                      style: TextStyle(
                          fontSize: 20,
                          color: secondaryColor),
                    ),
                  ),
                ),
              ],
            )
                        : Column(
                            children: [
                              sectionHeading("Platform Based Cleanliness"),

                              // Creating Cleanliness Pie Chart and Cleanliness Line Bars for different platforms
                              createCleanlinessDashboard(
                                  platform1CleanlinessValue,
                                  platform1CleanlinessColor,
                                  platform1CleanlinessRadius,
                                  platform2CleanlinessValue,
                                  platform2CleanlinessColor,
                                  platform2CleanlinessRadius,
                                  platform3CleanlinessValue,
                                  platform3CleanlinessColor,
                                  platform3CleanlinessRadius,
                                  platform4CleanlinessValue,
                                  platform4CleanlinessColor,
                                  platform4CleanlinessRadius,
                                  platform5CleanlinessValue,
                                  platform5CleanlinessColor,
                                  platform5CleanlinessRadius,
                                  platform6CleanlinessValue,
                                  platform6CleanlinessColor,
                                  platform6CleanlinessRadius),

                              sectionHeading("Cleanliness status"),

                              // Creating pie charts for different types of Cleanliness Status
                              createCleanlinessTypePieChart(
                                  littleBadStatusValue,
                                  littleBadColor,
                                  badStatusValue,
                                  badColor,
                                  worseStatusValue,
                                  worseColor),
                            ],
                          )
                    : title.toLowerCase() == "work monitoring"
                        ? totalWorkMonitoring == 0
                            ? Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                const SizedBox(
                  height: 50,
                ),
                Center(
                  child: Lottie.asset(
                      "assets/animations/no_data_available.json",
                      animate: true,
                      width: 350),
                ),
                Center(
                  child: Text(
                    title,
                    style: TextStyle(
                        fontSize: 24,
                        color: primaryColor,
                        fontWeight: FontWeight.bold),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Center(
                    child: Text(
                      "No Data Found!",
                      style: TextStyle(
                          fontSize: 20,
                          color: secondaryColor),
                    ),
                  ),
                ),
              ],
            )
                            : Column(
                                children: [
                                  sectionHeading(
                                      "Platform Based Work Monitoring"),

                                  // Creating Work Monitoring Pie Chart and Work Monitoring Line Bars for different platforms
                                  createWorkMonitoringDashboard(
                                      platform1WorkMonitoringValue,
                                      platform1WorkMonitoringColor,
                                      platform1WorkMonitoringRadius,
                                      platform2WorkMonitoringValue,
                                      platform2WorkMonitoringColor,
                                      platform2WorkMonitoringRadius,
                                      platform3WorkMonitoringValue,
                                      platform3WorkMonitoringColor,
                                      platform3WorkMonitoringRadius,
                                      platform4WorkMonitoringValue,
                                      platform4WorkMonitoringColor,
                                      platform4WorkMonitoringRadius,
                                      platform5WorkMonitoringValue,
                                      platform5WorkMonitoringColor,
                                      platform5WorkMonitoringRadius,
                                      platform6WorkMonitoringValue,
                                      platform6WorkMonitoringColor,
                                      platform6WorkMonitoringRadius),
                                ],
                              )
                        : title.toLowerCase() == "crowd management"
                            ? totalCrowdManagement == 0
                                ? Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                const SizedBox(
                  height: 50,
                ),
                Center(
                  child: Lottie.asset(
                      "assets/animations/no_data_available.json",
                      animate: true,
                      width: 350),
                ),
                Center(
                  child: Text(
                    title,
                    style: TextStyle(
                        fontSize: 24,
                        color: primaryColor,
                        fontWeight: FontWeight.bold),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Center(
                    child: Text(
                      "No Data Found!",
                      style: TextStyle(
                          fontSize: 20,
                          color: secondaryColor),
                    ),
                  ),
                ),
              ],
            )
                                : Column(
                                    children: [
                                      sectionHeading("Platform Based Crowd"),

                                      // Creating Crowd management Pie Chart and Crowd management Line Bars for different platforms
                                      createCrowdManagementDashboard(
                                          platform1CrowdManagementValue,
                                          platform1CrowdManagementColor,
                                          platform1CrowdManagementRadius,
                                          platform2CrowdManagementValue,
                                          platform2CrowdManagementColor,
                                          platform2CrowdManagementRadius,
                                          platform3CrowdManagementValue,
                                          platform3CrowdManagementColor,
                                          platform3CrowdManagementRadius,
                                          platform4CrowdManagementValue,
                                          platform4CrowdManagementColor,
                                          platform4CrowdManagementRadius,
                                          platform5CrowdManagementValue,
                                          platform5CrowdManagementColor,
                                          platform5CrowdManagementRadius,
                                          platform6CrowdManagementValue,
                                          platform6CrowdManagementColor,
                                          platform6CrowdManagementRadius),

                                      sectionHeading("Crowd status"),

                                      // Creating pie charts for different types of Crowd management status
                                      createCrowdTypePieChart(
                                          lessCrowdValue,
                                          lessCrowdColor,
                                          normalCrowdValue,
                                          normalCrowdColor,
                                          highCrowdValue,
                                          highCrowdColor,
                                          veryHighCrowdValue,
                                          veryHighCrowdColor),
                                    ],
                                  )
                            : Column(
                                mainAxisAlignment: MainAxisAlignment.start,
                                children: [
                                  const SizedBox(
                                    height: 50,
                                  ),
                                  Center(
                                    child: Lottie.asset(
                                        "assets/animations/no_data_available.json",
                                        animate: true,
                                        width: 350),
                                  ),
                                  Center(
                                    child: Text(
                                      title,
                                      style: TextStyle(
                                          fontSize: 24,
                                          color: primaryColor,
                                          fontWeight: FontWeight.bold),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Center(
                                      child: Text(
                                        "No Data Found!",
                                        style: TextStyle(
                                            fontSize: 20,
                                            color: secondaryColor),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
          ],
        ),
      ),
    );
  }
}
