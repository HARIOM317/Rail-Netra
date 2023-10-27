import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:path_provider/path_provider.dart';
import 'package:railway/src/utils/constants.dart';

import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

class ReportsPage extends StatefulWidget {
  const ReportsPage({super.key});

  @override
  State<ReportsPage> createState() => _ReportsPageState();
}

class _ReportsPageState extends State<ReportsPage> {
  String currentDate = DateFormat.d().format(DateTime.now());
  String currentMonth = DateFormat.M().format(DateTime.now());
  String currentYear = DateFormat.y().format(DateTime.now());

  double fontSize = 14.0;

  // required variables to button of top row
  String title = "CRIME";
  Color crimeButtonBackgroundColor = primaryColor;
  Color crimeButtonTextColor = Colors.white;

  Color cleanlinessButtonBackgroundColor = Colors.white;
  Color cleanlinessButtonTextColor = primaryColor;

  Color workMonitoringButtonBackgroundColor = Colors.white;
  Color workMonitoringButtonTextColor = primaryColor;

  Color crowdManagementButtonBackgroundColor = Colors.white;
  Color crowdManagementButtonTextColor = primaryColor;

  // To Generate PDF
  final pdf = pw.Document();
  File? file;
  DateTime time = DateTime.now();

  // Variables for today's data
  int totalCrimeDetected = 0;
  int totalWorkDetected = 0;
  int totalCrowdDetected = 0;
  int totalCleanlinessDetected = 0;

  // Variables for whole data
  var platform1Crime = 0;
  var platform2Crime = 0;
  var platform3Crime = 0;
  var platform4Crime = 0;
  var platform5Crime = 0;
  var platform6Crime = 0;

  var platform1Crowd = 0;
  var platform2Crowd = 0;
  var platform3Crowd = 0;
  var platform4Crowd = 0;
  var platform5Crowd = 0;
  var platform6Crowd = 0;

  var platform1Cleanliness = 0;
  var platform2Cleanliness = 0;
  var platform3Cleanliness = 0;
  var platform4Cleanliness = 0;
  var platform5Cleanliness = 0;
  var platform6Cleanliness = 0;

  var platform1Work = 0;
  var platform2Work = 0;
  var platform3Work = 0;
  var platform4Work = 0;
  var platform5Work = 0;
  var platform6Work = 0;

  final todayHeaders = ['Today', 'Activity'];
  final headers = ['Platform', 'Crime', 'Crowd', 'Cleanliness', 'Work'];

  writeOnPDF() async {
    final logo = pw.MemoryImage(
      (await rootBundle.load('assets/images/icons/logo.png'))
          .buffer
          .asUint8List(),
    );

    pdf.addPage(pw.Page(
        pageFormat: PdfPageFormat.a4,
        build: (pw.Context context) {
          return pw.Column(children: [
            // app logo and date
            pw.Padding(
              padding: const pw.EdgeInsets.all(15.0),
              child: pw.Row(
                mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                children: [
                  pw.Image(logo, width: 80),
                  pw.Text("Rail-Netra", style: pw.TextStyle(color: PdfColors.blue500, fontSize: 28, fontWeight: pw.FontWeight.bold)),
                  pw.Text(
                    "Date: $currentDate-$currentMonth-$currentYear",
                    style: pw.TextStyle(
                        fontSize: 16,
                        fontWeight: pw.FontWeight.bold,
                        color: PdfColors.blue500),
                  )
                ],
              ),
            ),

            pw.Align(
              alignment: pw.Alignment.topLeft,
              child: pw.Padding(
                  padding: const pw.EdgeInsets.symmetric(vertical: 10.0),
                  child: pw.Text("TODAY'S DATA",
                      style: pw.TextStyle(
                          color: PdfColors.blue,
                          fontSize: 16,
                          fontWeight: pw.FontWeight.bold))),
            ),

            // Table for today's data
            pw.TableHelper.fromTextArray(
              headers: todayHeaders,
              data: <List<String>>[
                <String>[
                  'Total Crime Cases',
                  '$totalCrimeDetected',
                ],
                <String>[
                  'Total Crowd Cases',
                  '$totalCrowdDetected',
                ],
                <String>[
                  'Total Cleanliness Cases',
                  '$totalCleanlinessDetected',
                ],
                <String>[
                  'Total Work Cases',
                  '$totalWorkDetected',
                ],
              ],
              headerStyle: pw.TextStyle(fontWeight: pw.FontWeight.bold, color: PdfColors.black, fontSize: 16),
              cellStyle: const pw.TextStyle(color: PdfColors.black, fontSize: 16),
              headerAlignment: pw.Alignment.centerLeft,
              cellAlignment: pw.Alignment.centerLeft,
              headerHeight: 40,
              cellHeight: 30,
              headerPadding: const pw.EdgeInsets.symmetric(horizontal: 10),
              cellPadding: const pw.EdgeInsets.symmetric(horizontal: 10),
            ),

            pw.SizedBox(height: 50),

            pw.Align(
              alignment: pw.Alignment.topLeft,
              child: pw.Padding(
                  padding: const pw.EdgeInsets.symmetric(vertical: 10.0),
                  child: pw.Text("WHOLE DATA",
                      style: pw.TextStyle(
                          color: PdfColors.blue,
                          fontSize: 16,
                          fontWeight: pw.FontWeight.bold))),
            ),
            // Table for whole data
            pw.TableHelper.fromTextArray(
                headers: headers,
                data: <List<String>>[
                  <String>[
                    'Platform 1',
                    '$platform1Crime',
                    '$platform1Crowd',
                    '$platform1Cleanliness',
                    '$platform1Work'
                  ],
                  <String>[
                    'Platform 2',
                    '$platform2Crime',
                    '$platform2Crowd',
                    '$platform2Cleanliness',
                    '$platform2Work'
                  ],
                  <String>[
                    'Platform 3',
                    '$platform3Crime',
                    '$platform3Crowd',
                    '$platform3Cleanliness',
                    '$platform3Work'
                  ],
                  <String>[
                    'Platform 4',
                    '$platform4Crime',
                    '$platform4Crowd',
                    '$platform4Cleanliness',
                    '$platform4Work'
                  ],
                  <String>[
                    'Platform 5',
                    '$platform5Crime',
                    '$platform5Crowd',
                    '$platform5Cleanliness',
                    '$platform5Work'
                  ],
                  <String>[
                    'Platform 6',
                    '$platform6Crime',
                    '$platform6Crowd',
                    '$platform6Cleanliness',
                    '$platform6Work'
                  ],
                ],
                headerStyle: pw.TextStyle(
                    fontWeight: pw.FontWeight.bold,
                    color: PdfColors.black,
                    fontSize: 16),
                cellStyle:
                    const pw.TextStyle(color: PdfColors.black, fontSize: 16),
                cellAlignment: pw.Alignment.center,
                headerHeight: 40,
                cellHeight: 30)
          ]);
        }));
  }

  Future savePDF() async {
    final directory = await getExternalStorageDirectory();
    if (kDebugMode) {
      print(directory?.path);
    }

    file = File(
        "${directory?.path}/Rail-Netra Report $currentYear-$currentMonth-$currentDate.pdf");

    final pdfBytes = await pdf.save();
    await file?.writeAsBytes(pdfBytes);
    // ignore: use_build_context_synchronously
    showSnackbar(context, "PDF Saved Successfully to File/Android/data/com.management.railway/files/Rail-Netra Report $currentYear-$currentMonth-$currentDate.pdf", bgColor: Colors.green);
  }

  // function to get today's activities
  void getTotalData() async {
    var query1 = FirebaseFirestore.instance
        .collection('crime_videos')
        .where('date', isEqualTo: "$currentYear-$currentMonth-$currentDate");
    var snapshot1 = await query1.get();
    totalCrimeDetected = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('cleanliness_videos')
        .where('date', isEqualTo: "$currentYear-$currentMonth-$currentDate");
    var snapshot2 = await query2.get();
    totalCleanlinessDetected = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('date', isEqualTo: "$currentYear-$currentMonth-$currentDate");
    var snapshot3 = await query3.get();
    totalCrowdDetected = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('date', isEqualTo: "$currentYear-$currentMonth-$currentDate");
    var snapshot4 = await query4.get();
    totalWorkDetected = snapshot4.size;
  }

  // function to get crime cases of all platforms
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
  }

  // function to get crowd cases of all platforms
  void getCrowd() async {
    var query1 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1Crowd = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2Crowd = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3Crowd = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4Crowd = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5Crowd = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('crowd_management_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6Crowd = snapshot6.size;
  }

  // function to get work cases of all platforms
  void getWork() async {
    var query1 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 1");
    var snapshot1 = await query1.get();
    platform1Work = snapshot1.size;

    var query2 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 2");
    var snapshot2 = await query2.get();
    platform2Work = snapshot2.size;

    var query3 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 3");
    var snapshot3 = await query3.get();
    platform3Work = snapshot3.size;

    var query4 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 4");
    var snapshot4 = await query4.get();
    platform4Work = snapshot4.size;

    var query5 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 5");
    var snapshot5 = await query5.get();
    platform5Work = snapshot5.size;

    var query6 = FirebaseFirestore.instance
        .collection('work_monitoring_videos')
        .where('platform', isEqualTo: "Platform 6");
    var snapshot6 = await query6.get();
    platform6Work = snapshot6.size;
  }

  // function to get cleanliness cases of all platforms
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
  }

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

  @override
  void initState() {
    super.initState();

    setState(() {
      getTotalData();
      getCrimes();
      getCrowd();
      getCleanliness();
      getWork();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
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
                      "$title REPORTS",
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

            // app logo and date
            Padding(
              padding: const EdgeInsets.all(15.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Image.asset("assets/images/icons/logo.png", width: 60),
                  Text(
                    "Date: $currentYear-$currentMonth-$currentDate",
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: secondaryColor),
                  )
                ],
              ),
            ),

            // table column
            title.toLowerCase() == "crime"
                ? Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(10.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              "Crime",
                              style: TextStyle(
                                  fontSize: fontSize,
                                  color: Colors.black,
                                  fontWeight: FontWeight.bold),
                            ),
                            Text(
                              "Platform",
                              style: TextStyle(
                                  fontSize: fontSize,
                                  color: Colors.black,
                                  fontWeight: FontWeight.bold),
                            ),
                            Text(
                              "Time",
                              style: TextStyle(
                                  fontSize: fontSize,
                                  color: Colors.black,
                                  fontWeight: FontWeight.bold),
                            ),
                            Text(
                              "Accuracy",
                              style: TextStyle(
                                  fontSize: fontSize,
                                  color: Colors.black,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10.0),
                        child: Container(
                          width: double.infinity,
                          height: 2,
                          color: Colors.black,
                        ),
                      ),
                    ],
                  )
                : title.toLowerCase() == "cleanliness"
                    ? Column(
                        children: [
                          Padding(
                            padding: const EdgeInsets.all(10.0),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  "Status",
                                  style: TextStyle(
                                      fontSize: fontSize,
                                      color: Colors.black,
                                      fontWeight: FontWeight.bold),
                                ),
                                Text(
                                  "Platform",
                                  style: TextStyle(
                                      fontSize: fontSize,
                                      color: Colors.black,
                                      fontWeight: FontWeight.bold),
                                ),
                                Text(
                                  "Time",
                                  style: TextStyle(
                                      fontSize: fontSize,
                                      color: Colors.black,
                                      fontWeight: FontWeight.bold),
                                ),
                                Text(
                                  "Accuracy",
                                  style: TextStyle(
                                      fontSize: fontSize,
                                      color: Colors.black,
                                      fontWeight: FontWeight.bold),
                                ),
                              ],
                            ),
                          ),
                          Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 10.0),
                            child: Container(
                              width: double.infinity,
                              height: 2,
                              color: Colors.black,
                            ),
                          ),
                        ],
                      )
                    : title.toLowerCase() == "work monitoring"
                        ? Column(
                            children: [
                              Padding(
                                padding: const EdgeInsets.all(10.0),
                                child: Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(
                                      "Status",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Platform",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Time",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Accuracy",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                  ],
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10.0),
                                child: Container(
                                  width: double.infinity,
                                  height: 2,
                                  color: Colors.black,
                                ),
                              ),
                            ],
                          )
                        : Column(
                            children: [
                              Padding(
                                padding: const EdgeInsets.all(10.0),
                                child: Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(
                                      "Estimate People",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Platform",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Time",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      "Accuracy",
                                      style: TextStyle(
                                          fontSize: fontSize,
                                          color: Colors.black,
                                          fontWeight: FontWeight.bold),
                                    ),
                                  ],
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10.0),
                                child: Container(
                                  width: double.infinity,
                                  height: 2,
                                  color: Colors.black,
                                ),
                              ),
                            ],
                          ),

            Expanded(
              child: StreamBuilder(
                stream: title.toLowerCase() == "crime"
                    ? FirebaseFirestore.instance
                        .collection('crime_videos')
                        .where('date',
                            isEqualTo:
                                "$currentYear-$currentMonth-$currentDate")
                        .snapshots()
                    : title.toLowerCase() == "cleanliness"
                        ? FirebaseFirestore.instance
                            .collection('cleanliness_videos')
                            .where('date',
                                isEqualTo:
                                    "$currentYear-$currentMonth-$currentDate")
                            .snapshots()
                        : title.toLowerCase() == "work monitoring"
                            ? FirebaseFirestore.instance
                                .collection('work_monitoring_videos')
                                .where('date',
                                    isEqualTo:
                                        "$currentYear-$currentMonth-$currentDate")
                                .snapshots()
                            : FirebaseFirestore.instance
                                .collection('crowd_management_videos')
                                .where('date',
                                    isEqualTo:
                                        "$currentYear-$currentMonth-$currentDate")
                                .snapshots(),
                builder: (BuildContext context,
                    AsyncSnapshot<QuerySnapshot> snapshot) {
                  if (!snapshot.hasData || snapshot.data?.size == 0) {
                    return Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        const SizedBox(
                          height: 100,
                        ),
                        Center(
                          child: Lottie.asset("assets/animations/reports.json",
                              animate: true, width: 250),
                        ),
                        Center(
                          child: Text(
                            "No ${title.toLowerCase()} reports!",
                            style: const TextStyle(
                                fontSize: 20, color: Colors.black54),
                          ),
                        ),
                      ],
                    );
                  }
                  return ListView.builder(
                      physics: const BouncingScrollPhysics(),
                      itemCount: snapshot.data!.docs.length,
                      itemBuilder: (BuildContext context, int index) {
                        final data = snapshot.data!.docs[index];

                        return title.toLowerCase() == "crime"
                            ? Column(
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10.0, vertical: 15),
                                    child: Row(
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceBetween,
                                      children: [
                                        Text(
                                          data['label'],
                                          style: TextStyle(
                                              fontSize: fontSize,
                                              color: Colors.black87),
                                        ),
                                        Text(
                                          data['platform'],
                                          style: TextStyle(
                                              fontSize: fontSize,
                                              color: Colors.black87),
                                        ),
                                        Text(
                                          data['time'],
                                          style: TextStyle(
                                              fontSize: fontSize,
                                              color: Colors.black87),
                                        ),
                                        Text(
                                          "${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                          style: TextStyle(
                                              fontSize: fontSize,
                                              color: Colors.black87),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10),
                                    child: Container(
                                      width: double.infinity,
                                      height: 1,
                                      color: Colors.black26,
                                    ),
                                  )
                                ],
                              )
                            : title.toLowerCase() == "cleanliness"
                                ? Column(
                                    children: [
                                      Padding(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 10.0, vertical: 15),
                                        child: Row(
                                          mainAxisAlignment:
                                              MainAxisAlignment.spaceBetween,
                                          children: [
                                            Text(
                                              data['status'],
                                              style: TextStyle(
                                                  fontSize: fontSize,
                                                  color: Colors.black87),
                                            ),
                                            Text(
                                              data['platform'],
                                              style: TextStyle(
                                                  fontSize: fontSize,
                                                  color: Colors.black87),
                                            ),
                                            Text(
                                              data['time'],
                                              style: TextStyle(
                                                  fontSize: fontSize,
                                                  color: Colors.black87),
                                            ),
                                            Text(
                                              "${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                              style: TextStyle(
                                                  fontSize: fontSize,
                                                  color: Colors.black87),
                                            ),
                                          ],
                                        ),
                                      ),
                                      Padding(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 10),
                                        child: Container(
                                          width: double.infinity,
                                          height: 1,
                                          color: Colors.black26,
                                        ),
                                      )
                                    ],
                                  )
                                : title.toLowerCase() == "work monitoring"
                                    ? Column(
                                        children: [
                                          Padding(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 10.0, vertical: 15),
                                            child: Row(
                                              mainAxisAlignment:
                                                  MainAxisAlignment
                                                      .spaceBetween,
                                              children: [
                                                Text(
                                                  data['label'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  data['platform'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  data['time'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  "${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                              ],
                                            ),
                                          ),
                                          Padding(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 10),
                                            child: Container(
                                              width: double.infinity,
                                              height: 1,
                                              color: Colors.black26,
                                            ),
                                          )
                                        ],
                                      )
                                    : Column(
                                        children: [
                                          Padding(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 10.0, vertical: 15),
                                            child: Row(
                                              mainAxisAlignment:
                                                  MainAxisAlignment
                                                      .spaceBetween,
                                              children: [
                                                Text(
                                                  data['label'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  data['platform'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  data['time'],
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                                Text(
                                                  "${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                                  style: TextStyle(
                                                      fontSize: fontSize,
                                                      color: Colors.black87),
                                                ),
                                              ],
                                            ),
                                          ),
                                          Padding(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 10),
                                            child: Container(
                                              width: double.infinity,
                                              height: 1,
                                              color: Colors.black26,
                                            ),
                                          )
                                        ],
                                      );
                      });
                },
              ),
            ),
          ],
        ),
      ),

      // Button to Download PDF
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.picture_as_pdf),
        onPressed: () async {
          writeOnPDF();
          savePDF();
        },
        label: const Text("Save PDF"),
      ),
    );
  }
}
