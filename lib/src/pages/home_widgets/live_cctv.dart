import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/components/custom_textfield.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:railway/src/utils/live_stream.dart';
// import 'package:regexed_validator/regexed_validator.dart';

class LiveCCTV extends StatefulWidget {
  const LiveCCTV({super.key});

  @override
  State<LiveCCTV> createState() => _LiveCCTVState();
}

class _LiveCCTVState extends State<LiveCCTV> {
  TextEditingController urlController = TextEditingController();
  TextEditingController platformController = TextEditingController();
  String? id;
  bool isSaving = false;

  bool isFiltered = false;
  String? filteredPlatform;

  static String? selectedPlatform;

  FirebaseAuth auth = FirebaseAuth.instance;

  // function to save cctv url or ip address into firebase database
  saveCCTVUrl() async {
    setState(() {
      isSaving = true;
    });

    // creating collection of cctv_urls
    await FirebaseFirestore.instance.collection('cctv_urls').add({
      'urlLink': urlController.text,
      'platform': selectedPlatform,
    }).then((value) {
      setState(() {
        isSaving = false;
        Fluttertoast.showToast(msg: "CCTV added successfully!");
      });
    });
  }

  // dialog box to add cctv url into firebase
  addCCTVDialog(BuildContext context) {
    showDialog(
        context: context,
        builder: (_) {
          return AlertDialog(
            contentPadding: const EdgeInsets.only(left: 10, right: 10),
            shape: const RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(20))),
            title: const Text(
              "Link a CCTV Camera",
              textAlign: TextAlign.center,
            ),
            content: Form(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(
                        vertical: 35.0, horizontal: 10),
                    child: LoginTextField(
                      controller: urlController,
                      hintText: "Enter URL or IP",
                      maxLines: 1,
                    ),
                  ),
                  platformDropdownList(
                      selectedPlatform: selectedPlatform,
                      callbackFunction: (String? newPlatform) {
                        setState(() {
                          selectedPlatform = newPlatform!;
                          Fluttertoast.showToast(
                              msg: "$selectedPlatform Selected");
                          Navigator.pop(context);
                          addCCTVDialog(context);
                        });
                      }),
                ],
              ),
            ),
            actions: [
              Container(
                margin: const EdgeInsets.all(20),
                height: 60,
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                      backgroundColor: primaryColor,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30))),
                  onPressed: selectedPlatform != null
                      ? () {
                          // if (validator.url(urlController.text) || validator.ip(urlController.text)) {
                          //   setState(() {
                          //     saveCCTVUrl();
                          //   });
                          //   Navigator.pop(context);
                          // } else {
                          //   Fluttertoast.showToast(msg: "Invalid URL or IP!");
                          // }

                          setState(() {
                            saveCCTVUrl();
                          });
                          Navigator.pop(context);
                        }
                      : null,
                  child: const Text(
                    "Link",
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                  ),
                ),
              )
            ],
          );
        });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: primaryColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          filteredPlatformDropdownList(
              selectedPlatform: filteredPlatform,
              callbackFunction: (String? newPlatform) async {
                setState(() {
                  isFiltered = true;
                  filteredPlatform = newPlatform!;
                });
              }),
        ],
      ),
      body: isSaving == true
          ? Center(
              child: CircularProgressIndicator(
              color: primaryColor,
              backgroundColor: primaryColor.withOpacity(0.5),
            ))
          : SafeArea(
              child: Column(
                children: [
                  Expanded(
                    child: StreamBuilder(
                      stream: isFiltered
                          ? FirebaseFirestore.instance
                              .collection('cctv_urls')
                              .where('platform', isEqualTo: filteredPlatform)
                              .snapshots()
                          : FirebaseFirestore.instance
                              .collection('cctv_urls')
                              .snapshots(),
                      builder: (BuildContext context,
                          AsyncSnapshot<QuerySnapshot> snapshot) {
                        if (!snapshot.hasData || snapshot.data?.size == 0) {
                          return Column(
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              const SizedBox(height: 100,),

                              Center(
                                child: Lottie.asset(
                                    "assets/animations/no_data_found.json",
                                    animate: true,
                                    width: 300),
                              ),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 20.0),
                                child: Center(
                                  child: Text(
                                    "No CCTV Camera Available!",
                                    style: TextStyle(
                                        fontSize: 20, color: primaryColor),
                                  ),
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
                              return isFiltered
                                  ? GestureDetector(
                                      onLongPress: () async {
                                        showDialog(
                                            context: context,
                                            builder: (BuildContext context) {
                                              return AlertDialog(
                                                elevation: 10,
                                                shadowColor: Colors.deepPurple
                                                    .withOpacity(0.25),
                                                backgroundColor: Colors.white,
                                                shape:
                                                    const RoundedRectangleBorder(
                                                  borderRadius:
                                                      BorderRadius.all(
                                                          Radius.circular(
                                                              20.0)),
                                                ),
                                                title: Text(
                                                  'Remove CCTV',
                                                  style: TextStyle(
                                                    color: primaryColor,
                                                  ),
                                                ),
                                                content: const Text(
                                                  'Are you sure to remove CCTV?',
                                                  style: TextStyle(
                                                      color: Colors.black87),
                                                ),
                                                actions: <Widget>[
                                                  OutlinedButton(
                                                      onPressed: () {
                                                        Navigator.of(context)
                                                            .pop(false);
                                                      },
                                                      style: ElevatedButton
                                                          .styleFrom(
                                                        backgroundColor:
                                                            Colors.green[500],
                                                        shape: RoundedRectangleBorder(
                                                            borderRadius:
                                                                BorderRadius
                                                                    .circular(
                                                                        10)),
                                                        foregroundColor:
                                                            Colors.white,
                                                      ),
                                                      child:
                                                          const Text("Cancel")),
                                                  ElevatedButton(
                                                    onPressed: () async {
                                                      QuerySnapshot
                                                          querySnapshot =
                                                          await FirebaseFirestore
                                                              .instance
                                                              .collection(
                                                                  'cctv_urls')
                                                              .where('urlLink',
                                                                  isEqualTo: data[
                                                                      'urlLink'])
                                                              .get();

                                                      for (var doc
                                                          in querySnapshot
                                                              .docs) {
                                                        doc.reference.delete();
                                                      }

                                                      // ignore: use_build_context_synchronously
                                                      Navigator.of(context)
                                                          .pop(false);
                                                    },
                                                    style: ElevatedButton
                                                        .styleFrom(
                                                      backgroundColor:
                                                          Colors.red[500],
                                                      shape:
                                                          RoundedRectangleBorder(
                                                              borderRadius:
                                                                  BorderRadius
                                                                      .circular(
                                                                          10)),
                                                      foregroundColor:
                                                          Colors.white,
                                                    ),
                                                    child: const Text("Remove"),
                                                  ),
                                                ],
                                              );
                                            });
                                      },
                                      child: LiveStream(url: data['urlLink'] + "/video"))
                                  : Stack(
                                      children: [
                                        GestureDetector(
                                            onLongPress: () async {
                                              showDialog(
                                                  context: context,
                                                  builder:
                                                      (BuildContext context) {
                                                    return AlertDialog(
                                                      elevation: 10,
                                                      shadowColor: Colors
                                                          .black54
                                                          .withOpacity(0.25),
                                                      backgroundColor:
                                                          Colors.white,
                                                      shape:
                                                          const RoundedRectangleBorder(
                                                        borderRadius:
                                                            BorderRadius.all(
                                                                Radius.circular(
                                                                    20.0)),
                                                      ),
                                                      title: Text(
                                                        'Remove CCTV',
                                                        style: TextStyle(
                                                          color: primaryColor,
                                                        ),
                                                      ),
                                                      content: const Text(
                                                        'Are you sure to remove CCTV?',
                                                        style: TextStyle(
                                                            color:
                                                                Colors.black87),
                                                      ),
                                                      actions: <Widget>[
                                                        OutlinedButton(
                                                            onPressed: () {
                                                              Navigator.of(
                                                                      context)
                                                                  .pop(false);
                                                            },
                                                            style:
                                                                ElevatedButton
                                                                    .styleFrom(
                                                              backgroundColor:
                                                                  Colors.green[
                                                                      500],
                                                              shape: RoundedRectangleBorder(
                                                                  borderRadius:
                                                                      BorderRadius
                                                                          .circular(
                                                                              10)),
                                                              foregroundColor:
                                                                  Colors.white,
                                                            ),
                                                            child: const Text(
                                                                "Cancel")),
                                                        ElevatedButton(
                                                          onPressed: () async {
                                                            QuerySnapshot
                                                                querySnapshot =
                                                                await FirebaseFirestore
                                                                    .instance
                                                                    .collection(
                                                                        'cctv_urls')
                                                                    .where(
                                                                        'urlLink',
                                                                        isEqualTo:
                                                                            data['urlLink'])
                                                                    .get();

                                                            for (var doc
                                                                in querySnapshot
                                                                    .docs) {
                                                              doc.reference
                                                                  .delete();
                                                            }

                                                            // ignore: use_build_context_synchronously
                                                            Navigator.of(
                                                                    context)
                                                                .pop(false);
                                                          },
                                                          style: ElevatedButton
                                                              .styleFrom(
                                                            backgroundColor:
                                                                Colors.red[500],
                                                            shape: RoundedRectangleBorder(
                                                                borderRadius:
                                                                    BorderRadius
                                                                        .circular(
                                                                            10)),
                                                            foregroundColor:
                                                                Colors.white,
                                                          ),
                                                          child: const Text(
                                                              "Remove"),
                                                        ),
                                                      ],
                                                    );
                                                  });
                                            },
                                            child: LiveStream(
                                                url: data['urlLink'] +
                                                    "/video")),
                                        Positioned(
                                            top: 15,
                                            left: 15,
                                            child: Text(
                                              data['platform'],
                                              style: const TextStyle(
                                                  color: Colors.white),
                                            ))
                                      ],
                                    );
                            });
                      },
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Align(
                      alignment: FractionalOffset.bottomRight,
                      child: FloatingActionButton.extended(
                        onPressed: () {
                          addCCTVDialog(context);
                        },
                        icon: Image.asset(
                          "assets/images/icons/drawer_icons/live_drawer.png",
                          height: 24,
                        ),
                        label: const Text("Add CCTV"),
                      ),
                    ),
                  )
                ],
              ),
            ),
    );
  }
}
