package de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.mkapp;

import java.util.Arrays;

import de.lmu.ifi.dbs.data.DatabaseObject;
import de.lmu.ifi.dbs.distance.NumberDistance;
import de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.AbstractMTree;
import de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.AbstractMTreeNode;
import de.lmu.ifi.dbs.persistent.PageFile;
import de.lmu.ifi.dbs.utilities.output.Format;

/**
 * Represents a node in an MkApp-Tree.
 *
 * @author Elke Achtert (<a href="mailto:achtert@dbs.ifi.lmu.de">achtert@dbs.ifi.lmu.de</a>)
 */
class MkAppTreeNode<O extends DatabaseObject, D extends NumberDistance<D>> extends AbstractMTreeNode<O, D, MkAppTreeNode<O, D>, MkAppEntry<D>> {
  /**
   * Empty constructor for Externalizable interface.
   */
  public MkAppTreeNode() {
	  // empty constructor
  }

  /**
   * Creates a MkAppTreeNode object.
   *
   * @param file     the file storing the MCop-Tree
   * @param capacity the capacity (maximum number of entries plus 1 for overflow) of this node
   * @param isLeaf   indicates wether this node is a leaf node
   */
  public MkAppTreeNode(PageFile<MkAppTreeNode<O, D>> file, int capacity, boolean isLeaf) {
    super(file, capacity, isLeaf);
  }

  /**
   * Creates a new leaf node with the specified capacity.
   *
   * @param capacity the capacity of the new node
   * @return a new leaf node
   */
  protected MkAppTreeNode<O, D> createNewLeafNode(int capacity) {
    return new MkAppTreeNode<O, D>(getFile(), capacity, true);
  }

  /**
   * Creates a new directory node with the specified capacity.
   *
   * @param capacity the capacity of the new node
   * @return a new directory node
   */
  protected MkAppTreeNode<O, D> createNewDirectoryNode(int capacity) {
    return new MkAppTreeNode<O, D>(getFile(), capacity, false);
  }

  /**
   * Determines and returns the polynomial approximation for the knn distances of this node
   * as the maximum of the polynomial approximations of all entries.
   *
   * @return the conservative approximation for the knn distances
   */
  protected PolynomialApproximation knnDistanceApproximation() {
    int p_max = 0;
    double[] b = null;
    for (int i = 0; i < getNumEntries(); i++) {
      MkAppEntry<D> entry = getEntry(i);
      PolynomialApproximation approximation = entry.getKnnDistanceApproximation();
      if (b == null) {
        p_max = approximation.getPolynomialOrder();
        b = new double[p_max];
      }
      for (int p = 0; p < p_max; p++) {
        b[p] += approximation.getB(p);
      }
    }

    for (int p = 0; p < p_max; p++) {
      b[p] /= p_max;
    }

    if (debug) {
      StringBuffer msg = new StringBuffer();
      msg.append("b " + Format.format(b, 4));
      debugFine(msg.toString());
    }

    return new PolynomialApproximation(b);
  }

  /**
   * Adjusts the parameters of the entry representing this node.
   *
   * @param entry           the entry representing this node
   * @param routingObjectID the id of the (new) routing object of this node
   * @param parentDistance  the distance from the routing object of this node
   *                        to the routing object of the parent node
   * @param mTree           the M-Tree object holding this node
   */
  public void adjustEntry(MkAppEntry<D> entry, Integer routingObjectID, D parentDistance, AbstractMTree<O, D, MkAppTreeNode<O, D>, MkAppEntry<D>> mTree) {
    super.adjustEntry(entry, routingObjectID, parentDistance, mTree);
//    entry.setKnnDistanceApproximation(knnDistanceApproximation());
  }

  /**
   * @see de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.AbstractMTreeNode#test(de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.MTreeEntry, de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.AbstractMTreeNode, int, de.lmu.ifi.dbs.index.tree.metrical.mtreevariants.AbstractMTree)
   */
  protected void test(MkAppEntry<D> parentEntry, MkAppTreeNode<O, D> parent, int index, AbstractMTree<O, D, MkAppTreeNode<O, D>, MkAppEntry<D>> mTree) {
    super.test(parentEntry, parent, index, mTree);

    MkAppEntry<D> entry = parent.getEntry(index);
    PolynomialApproximation approximation_soll = knnDistanceApproximation();
    PolynomialApproximation approximation_ist = entry.getKnnDistanceApproximation();

    if ( ! Arrays.equals(approximation_ist.getCoefficients(), approximation_soll.getCoefficients())) {
      String soll = approximation_soll.toString();
      String ist = entry.getKnnDistanceApproximation().toString();
       throw new RuntimeException("Wrong polynomial approximation in node "
                                  + parent.getID() + " at index " + index + " (child "
                                  + entry.getID() + ")" + "\nsoll: " + soll
                                  + ",\n ist: " + ist);

    }

  }
}
